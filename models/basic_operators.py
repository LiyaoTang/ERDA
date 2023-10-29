import os, re, sys
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from .utils import tf_scope

def masked_softmax(vec, valid_mask, axis=-1):
    if valid_mask.dtype == tf.bool:
        valid_mask = tf.cast(valid_mask, vec.dtype)
    assert valid_mask.dtype == vec.dtype, f'not supported mask {valid_mask.dtype} with vec {vec.dtype}'
    return tf.nn.softmax(vec - (1 - valid_mask) * _inf, axis=axis)

def dense_masked_softmax(logits, mask, T):
    """ Masked softmax over dim 1, mask broadcasts over dim 2, using normal softmax

    Args:
        logits: (N, L, T)
        mask: (N, L)
        T: number of dim 2
    Returns:
        probabilities (N, L, T)
    """

    v = T
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(np.array([[_inf]], dtype=np.float32), dtype=tf.float32)
    infs = tf.tile(inf, [tf.shape(indices)[0], v])
    infmask = tf.scatter_nd(
        indices=indices,
        updates=infs,
        shape=tf.shape(logits))
    _p = tf.nn.softmax(logits - infmask, axis=1)
    return _p


def sparse_masked_softmax(logits, mask):
    """Masked softmax over dim -1 using sparse softmax

    Args:
        logits: (N, L, T)
        mask: (N, L, T)

    Returns:
        probabilities (N, L, T)
    """

    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)
    return result


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd, stddev=1e-3, init='xavier'):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float.
      stddev: standard deviation of a truncated Gaussian
      init: weight initializer type

    Returns:
      Variable Tensor
    """
    if init == 'xavier':
        initializer = tf.glorot_uniform_initializer()
    elif init == 'msra':
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif init == 'fan_in':
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)
    elif not isinstance(init, str):
        initializer = init
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd > 0:  # L2 regularization
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        tf.add_to_collection('weight_losses', weight_decay)
    return var


def batch_norm(inputs, is_training, scope, bn_decay=0.99, epsilon=0.001):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    return tf.layers.batch_normalization(inputs, axis=-1,
                                         momentum=bn_decay,
                                         epsilon=epsilon,
                                         training=is_training,
                                         trainable=True,
                                         name=scope,
                                         fused=False)


def ind_max_pool(x, inds, scope):
    """ This tensorflow operation compute a maxpooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
        scope: name scope

    Returns:
        [n2, d] pooled features matrix
    """
    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.reduce_min(x, axis=0, keepdims=True)], axis=0)
        # Get features for each pooling cell [n2, max_num, d]
        pool_features = tf.gather(x, inds, axis=0)
        # Pool the maximum
        return tf.reduce_max(pool_features, axis=1)


def ind_closest_pool(x, inds, scope):
    """This tensorflow operation compute a pooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
        scope:

    Returns:
        [n2, d] pooled features matrix
    """

    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)
        # Get features for each pooling cell [n2, d]
        pool_features = tf.gather(x, inds[:, 0], axis=0)
        return pool_features


def conv1d_1x1(features,
               out_fdim,
               scope,
               is_training,
               with_bias=False,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """A simple 1x1 1D convolution

    Args:
        features: Input features, float32[n_points, in_fdim]
        out_fdim: Output features dim
        scope: name scope
        is_training: True indicates training phase
        with_bias: If True, adds a learnable bias to the output
        init: Weight initializer
        weight_decay: If > 0 , add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution


    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.matmul(features, w) + biases
        else:
            x = tf.matmul(features, w)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

def batch_conv1d_1x1(features,
                     out_fdim,
                     scope,
                     is_training,
                     with_bias=False,
                     init='xavier',
                     weight_decay=0,
                     activation_fn='relu',
                     bn=True,
                     bn_momentum=0.98,
                     bn_eps=1e-3):
    """A simple 1x1 1D convolution for batch inputs

        Args:
            features: Input features, float32[b, n_points, in_fdim]
            out_fdim: Output features dim
            scope: name scope
            is_training: True indicates training phase
            with_bias: If True, adds a learnable bias to the output
            init: Weight initializer
            weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
            activation_fn: Activation function
            bn: If True, add batch norm after convolution


        Returns:
            [b, n_points, out_fdim]
        """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.tensordot(features, w, 1) + biases
        else:
            x = tf.tensordot(features, w, 1)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x


def global_average_block(inputs, features, scope):
    """This Block performing a global average pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]

    """
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]
        # Add a last zero features for shadow batch inds
        features = tf.concat([features, tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)
        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)
        # Average features in each batch
        batch_features = tf.reduce_sum(batch_features, axis=1)
        # batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] >= 0, tf.float32), axis=1, keepdims=True)
        batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] < N, tf.float32), axis=1, keepdims=True)
        features = batch_features / batch_num
    return features


def global_max_block(inputs, features, scope):
    """This Block performing a global max pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]
    """

    # Average pooling to aggregate feature in the end
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]

        # Add a last zero features for shadow batch inds
        features = tf.concat([features, -256.0 + tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)

        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)

        # Average features in each batch
        batch_features = tf.reduce_max(batch_features, axis=1)

        features = batch_features

    return features

# global config
_eps = 1e-12
_inf = 1e9
def get_initializer(k):
    if not isinstance(k, str) and k is not None : return k  # assume to be an initialization already
    elif k in [None, 'xavier', 'glorot']: return tf.initializers.glorot_uniform()  # scale = 1.0, fan_avg (uniform is used in paper)
    elif k == 'glorotN': return tf.initializers.glorot_normal()  # scale = 1.0, fan_avg
    elif k.startswith('normalf'):  # normal full - usually for dense - seems better
        stddev = k[len('normalf'):]; stddev = float(stddev) if stddev else 1e-2; return tf.initializers.random_normal(stddev=stddev)
    elif k.startswith('normal'):  # calling tf.random.truncated_normal, trunc to [-2, 2]
        stddev = k[len('normal'):]; stddev = float(stddev) if stddev else 1e-2; return tf.initializers.truncated_normal(stddev=stddev)
    elif k == 'fanin': return tf.initializers.variance_scaling(scale=1.0)  # fan_in, tunc normal - default variance_scaling - usually for conv
    elif k == 'fanout': return tf.initializers.variance_scaling(scale=1.0, mode='fan_out')  # fan_out, tunc normal
    elif k in ['msra', 'he']: return tf.initializers.variance_scaling(scale=2.0)  # fan_in, tunc normal - he normal (as used in paper)
    elif k == 'heout': return tf.initializers.variance_scaling(scale=2.0, mode='fan_out')  # fan_out, tunc normal
    elif k == 'zeros': return tf.initializers.zeros()
    elif k == 'ones': return tf.initializers.ones()
    else:
        raise KeyError(f'not supported initializer = {k}')

def get_type(k):
    if k in [True, None, float, 'float', 'float32']: return tf.float32
    elif k in ['float16']: return tf.float16
    elif k in [int, 'int', 'int32']: return tf.int32
    elif k in [bool, 'bool']: return tf.bool
    elif hasattr(tf, k) and isinstance(k, str): return getattr(tf, k)
    else: return k

def tf_get_variable(name, shape, initializer=None, wd=None, device='/cpu:0'):
    initializer = get_initializer(initializer)
    with tf.device(device):
        var = tf.get_variable(name, shape=shape, initializer=initializer)  # use_resource=True

    if wd and wd > 0:  # L2 regularization
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('weight_losses', weight_decay)
    return var

def tf_gather(supports, neighbor_idx, shadow_fn=tf.zeros, get_mask=True):
    mask = None
    batch_dims = len(neighbor_idx.shape) - 2  # [..., N, k]
    # assert batch_dims >= 0, f'batch_dims = {batch_dims}, consider direct calling of tf.gather on supports = {supports}'
    if batch_dims == 0 and get_mask:  # radius search - [BxN, k]
        T = get_type(get_mask)  # may specify types
        N = tf.shape(supports)[0]
        mask = neighbor_idx < N if T == tf.bool else tf.cast(neighbor_idx < N, T)

    # supports - [..., N] or [..., N, d]
    if batch_dims == 0:
        if isinstance(shadow_fn, (int, float)):
            if shadow_fn == 0:
                shadow_fn = tf.zeros
            else:
                shadow_fn = lambda *args, _const=shadow_fn, **kwargs: tf.zeros(*args, **kwargs) + _const
        elif isinstance(shadow_fn, str):
            shadow_fn = lambda *args, _func=shadow_fn, **kwargs: getattr(tf, f'reduce_{_func}')(supports, axis=0, keepdims=True)
        if shadow_fn is not None:
            if isinstance(shadow_fn, tf.Tensor):
                shadow = shadow_fn
            else:
                shape = tf.shape(supports[:1, ...])  # [1, ...]
                shadow = shadow_fn(shape=shape, dtype=supports.dtype)
            supports = tf.concat([supports, shadow], axis=0)
    # batch_dims = max(batch_dims, 0)  # avoid '-1'
    neighbors = tf.gather(supports, neighbor_idx, batch_dims=batch_dims)

    rtn = (neighbors, mask) if get_mask else neighbors
    return rtn

def tf_cond(pred, true_fn=None, false_fn=None, name=None):
    from tensorflow.python.framework import smart_cond as smart_module

    """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

    If `pred` is a bool or has a constant value, we return either `true_fn()`
    or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

    Arguments:
        pred: A scalar determining whether to return the result of `true_fn` or `false_fn`.
        true_fn: The callable to be performed if pred is true.
        false_fn: The callable to be performed if pred is false.
        name: Optional name prefix when using `tf.cond`.

    Returns:
        Tensors returned by the call to either `true_fn` or `false_fn`.

    Raises:
        TypeError: If `true_fn` or `false_fn` is not callable.
    """
    if isinstance(pred, tf.Variable):
        return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)

def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation.startswith('lrelu') or activation.startswith('leaky_relu'):
            alpha = activation.split('relu')[-1]
            alpha = float(alpha) if alpha else 0.2
            activation = lambda x, a=alpha: tf.nn.leaky_relu(x, a)
        else:
            raise NotImplementedError(f'not supported activation = {activation}')

    return activation

def dense_layer(features, d_out, name=None, activation=None, bias=True, bn=True, is_training=None, initializer=None,
                weight_decay=0, bn_momentum=0.99, bn_eps=1e-6, **kwargs):
    if dense_layer.config and dense_layer.config.dense_by_conv:
        bias = False if bn else bias
        return conv1d_1x1(features, d_out, name, is_training, bias, initializer, weight_decay, activation, bn, bn_momentum, bn_eps)
    if name:
        with tf.variable_scope(name) as sc:
            return dense_layer(features, d_out, None, activation, bias, bn, is_training, initializer, weight_decay, bn_momentum, bn_eps)

    bn = 'bn' if bn is True else bn  # default to bn
    kernel = tf_get_variable('weights', shape=(int(features.shape[-1]), d_out,), initializer=initializer, wd=weight_decay)
    features = tf.tensordot(features, kernel, 1)
    if bn == 'bn':
        assert is_training is not None
        features = tf.layers.batch_normalization(features, momentum=bn_momentum, epsilon=bn_eps, training=is_training, fused=False)
    elif bn:
        raise NotImplementedError(f'not implemented norm = {bn}')
    elif bias:
        bias = tf_get_variable('bias', shape=(d_out,), initializer=tf.initializers.zeros())
        features = features + bias

    activation = get_activation(activation)
    if activation is not None:
        features = activation(features)
    return features
dense_layer.config = None  # config-dependent analysis use

@tf_scope
def mlps_by_ops(features, d_out, num_classes, ops, kwargs, **_kwargs):
    # warpper of mlps - specifying with str (ops)
    ops = f'{ops}mlp' if isinstance(ops, int) else ops
    if 'linear' not in ops: ops = ops.replace('lin', 'linear')
    if not (re.fullmatch('mlp\d*|\d*mlp', ops) or ops in ['linear', 'linearmap', 'linearbn']):
        raise ValueError(f'invalid ops = {ops}')  # e.g. 2mlp, mlp, mlp2, linear
    linear = not ops.endswith('mlp')  # 2mlp / mlp -> no linear        
    mlp_num = int(ops.replace('mlp', '')) if ops.replace('mlp', '').isdigit() else 1
    kwargs = kwargs.copy()
    kwargs.update(**_kwargs)
    if ops in ['linearmap', 'linearbn']:
        kwargs.update(linearbn={'linearmap': False, 'linearbn': True}[ops])
    features = mlps(features, d_out, num_classes, mlp_num, linear=linear, **kwargs)
    return features

@tf_scope
def mlps(features, d_out, num_classes, num_layers, is_training, dp=False, linear=True, bn=True, activation=tf.nn.leaky_relu,
         initializer=None, weight_decay=0, bn_momentum=0.99, bn_eps=1e-6, linearbn=False, _cnt=1, drop=None, **kwargs):
    if dp:
        raise NotImplementedError
    if isinstance(num_layers, str):
        assert num_layers.startswith('mlp') or num_layers == 'linear', f'not supported num_layers = {num_layers}'
        assert num_layers != 'linear' or linear  # num_layers='linear' -> must have linear=True
        num_layers = int(num_layers[len('mlp'):]) if num_layers.startswith('mlp') and num_layers[len('mlp'):] else 1

    for cnt in range(_cnt, num_layers + _cnt - int(linear)):  # if linear - last mlp becomes linear, else add activation to all mlp(s)
        features = dense_layer(features, d_out, f'mlp_{cnt}', activation, True, bn, is_training, initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps, **kwargs)

    if drop:
        drop = float(re.search('\.\d+', drop).group().group()) if isinstance(drop, str) else float(drop)
        assert 0 < drop and drop < 1
        features = dropout(features, rate=drop, is_training=is_training, name='dropout')

    if linear:
        l_bn = linearbn  # if linearbn is not None else bn
        logits = dense_layer(features, num_classes, f'linear', None, True, l_bn, is_training, initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps, **kwargs)
    else:
        logits = features
    return logits

def reduce_neighbors(features, neighbor_idx, reduction):
    if reduction == 'mean':
        features, mask = tf_gather(features, neighbor_idx)  # [B x sample_num, kr, d]
        features = tf.reduce_sum(features, axis=-2) / tf.reduce_sum(mask, axis=-1, keepdims=True) if mask is not None else tf.reduce_mean(features, -2)
    elif reduction == 'max':
        features, mask = tf_gather(features, neighbor_idx, shadow_fn=tf.reduce_min(features, axis=0, keepdims=True))
        features = tf.reduce_max(features, axis=-2)  # [B x sample_num, d]
    else:
        raise NotImplementedError(f'not support reduction = {reduction}')
    return features


def position_encode(center_xyz, neighbor_xyz, norm=None, encoding='dp', rtn='tensor', **kwargs):
    """
    Args:
        center_xyz/neighbor_xyz : [BxN, (1,) 3]/[BxN, k, 3] - xyz of center/neighboring points
    Returns:
        position encoding : [BxN, k, dim]
    """
    if center_xyz != 0 and len(center_xyz.shape) < len(neighbor_xyz.shape):  # align axis
        center_xyz = tf.expand_dims(center_xyz, axis=-2)

    # prepare potentially shared ops
    encoding = encoding.split('-')
    assert len(encoding) > 0, f'empty encoding: {encoding}'
    if any([enc in encoding for enc in ['dp', 'dpn', 'd', 'dn', 'r', 'dp2', 'dpn2', 'd2', 'dn2']]):  # relative xyz
        dp = neighbor_xyz - center_xyz  # [BxN, 3]
    if any([enc in encoding for enc in ['d', 'dn', 'r', 'd2', 'dn2']]):  # squared dist
        dist2 = tf.reduce_sum(dp ** 2, axis=-1, keepdims=True)
    if any([enc in encoding for enc in ['d', 'dn', 'r']]):  # dist
        dist = tf.sqrt(dist2 + _eps)
    if 'p' in encoding:  # p = pi, pj
        encoding.remove('p'); encoding += ['pi', 'pj']

    # collect desired encoding
    enc_dict = {}
    for enc in sorted(encoding):
        if enc == 'dp':
            enc_dict['dp'] = dp
        elif enc == 'dpn':
            enc_dict['dp_norm'] = dp / norm
        elif enc == 'pi':
            multiples = [1] * (len(center_xyz.shape) - 2) + [tf.shape(neighbor_xyz)[-2], 1]  # [BxN, ...] / [B, N, ...]
            pi = tf.tile(center_xyz, multiples)
            enc_dict['pi'] = pi
        elif enc == 'pj':
            enc_dict['pj'] = neighbor_xyz
        elif enc == 'd':
            enc_dict['dist'] = dist
        elif enc == 'dn':
            enc_dict['dist_norm'] = dist / norm

        elif enc == 'dp2':
            enc_dict['dp2'] = dp ** 2
        elif enc == 'dpn2':
            enc_dict['dpn2'] = dp ** 2 / norm ** 2
        elif enc == 'd2':
            enc_dict['dist2'] = dist2
        elif enc == 'dn2':
            enc_dict['dist_norm2'] = dist2 / norm ** 2

        elif enc == 'r':
            dp_flat = tf.reshape(dp, [-1, 3])
            # TODO: verify the need of _eps
            dist_xy = tf.reshape(tf.sqrt(dp_flat[:, 0]** 2 + dp_flat[:, 1]** 2 + _eps), tf.shape(dist))
            phi = tf.math.acos(x / dist_xy)
            theta = tf.math.acos(dist / center_xyz)
            enc_dict.update({'phi': phi, 'theta': theta, 'dist': dist})
        else:
            raise NotImplementedError(f'not supported position enc = {enc} in {encoding}')
    # flatten/concat if necessary
    if rtn in ['list', 'tensor']:
        pos_enc = [enc_dict[k] for k in sorted(enc_dict.keys())]
        if rtn == 'tensor':
            pos_enc = tf.concat(pos_enc, axis=-1) if len(pos_enc) > 1 else pos_enc[0]
    elif rtn == 'dict':
        pos_enc = enc_dict
    else:
        raise NotImplementedError(f'not supported return type: rtn = {rtn}')
    return pos_enc
# exposed valid positional encoding (pi, pj not exposed)
position_encode.encoding = ['dp', 'dpn', 'p', 'pi', 'pj', 'd', 'dn', 'r', 'dp2', 'dpn2', 'd2', 'dn2']

def position_encoding_transform(ops, d_out, pos_enc=None, center_xyz=None, neighbor_xyz=None, norm=None, **kwargs):
    if pos_enc is None:
        encoding = [i for i in ops.split('-') if i in position_encode.encoding]
        assert len(encoding) > 0, f'empty encoding from ops = {ops}'
        pos_enc = position_encode(center_xyz, neighbor_xyz, norm, '-'.join(encoding))
    ops = ops.split('-')[0]
    if 'lin' in ops or 'mlp' in ops:
        f_xyz = mlps_by_ops(pos_enc, d_out, d_out, ops=ops, kwargs=kwargs, name='pos')
    else:
        f_xyz = pos_enc
    return f_xyz

def relative_encoding(encoding, features, neighbor_features, center_xyz, neighbor_xyz, norm, d_in, kwargs_mlp):
    """ encoding to generate kernel for center-neighbor aggregation - spatial & feature """
    # prepare kernel encoding specification
    enc_spec = encoding
    f_list = []
    pos_list = []
    # enc = block_cfg.enc.split('-')
    enc = enc_spec.split('-')
    enc_mlp = enc[0] if 'linear' in enc[0] or 'mlp' in enc[0] else ''
    encoding = enc[1:] if enc_mlp else enc
    for c in encoding:
        if c in position_encode.encoding:
            pos_list.append(c)
        elif c in ['fi', 'fj']:
            f_list.append(f'p{c[1]}')
        elif c == 'df':
            f_list.append('dp')
        elif c == 'fd':  # feature dist
            f_list.append('d')
        else:
            # raise NotImplementedError(f'not supported kernel encoding {c} in {block_cfg.enc}')
            raise NotImplementedError(f'not supported kernel encoding {c} in {enc_spec}')
    assert len(pos_list) > 0 or len(f_list) > 0

    # apply tf ops
    kernel_f = []
    with tf.variable_scope('position_encoding'):
        if pos_list:
            encoding_ops = '-'.join([enc_mlp] + pos_list)
            f_xyz = position_encoding_transform(encoding_ops, d_in, None, center_xyz, neighbor_xyz, norm=norm, **kwargs_mlp)
            kernel_f.append(f_xyz)
    with tf.variable_scope('feature_encoding'):
        if f_list:
            kernel_f += position_encode(features, neighbor_features, '-'.join(f_list), rtn='list')
    return kernel_f

def check_mask(x, mask):
    """ check & try fix the mask to match the shape of x
    """
    if mask is not None and len(mask.shape) < len(x.shape):
        mask = tf.expand_dims(mask, axis=-1)
    if mask is not None:
        assert len(mask.shape) == len(x.shape), f'unmatched shape: mask.shape={mask.shape} v.s. x.shape={x.shape}'
    return mask

def join_features(features, f_sc, join):
    if join == 'concat':
        features = tf.concat([features, f_sc], axis=-1)
    elif join == 'sum':
        features += f_sc
    else:
        raise NotImplementedError(f'not implemented join type = {join}')
    return features

def normalize(x, norm_cfg, axis=None, mask=None):
    """
    Args:
        x   : [..., N, k, d]  - assume the last two axes being the spatial-feature dim
        mask: [..., N, k, /d] - valid mask
    """
    assert norm_cfg, f'norm_cfg = {norm_cfg}, should check for empty config outside the func'
    mask = check_mask(x, mask)

    _axis = axis
    for n in norm_cfg.split('-'):
        if _axis is None and n in normalize.channel: axis = -1
        elif _axis is None and n in normalize.spatial: axis = -2
        elif _axis is None and n in normalize.others :axis = -3

        if n == 'l2':  # channel
            unit = x * mask if mask is not None else x
            unit = tf.sqrt(tf.reduce_sum(unit ** 2, axis=axis, keepdims=True) + _eps)
            x = x / unit
        elif n == 'l1':  # channel
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(tf.abs(unit), axis=axis, keepdims=True) + _eps
            x = x / unit
        elif n == 'softmax':  # spatial
            x = masked_softmax(x, mask, axis=axis, impl='inf') if mask is not None else tf.nn.softmax(x, axis=axis)
        elif n == 'norm':  # spatial
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(unit, axis=axis, keepdims=True) + _eps  # potential sum to 0
            x = x / unit
        elif n == 'norml1':  # spatial - l1 (reduce to 'norm' if all positive)
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(tf.abs(unit), axis=axis, keepdims=True) + _eps
            x = x / unit
        elif n == 'G_softmax':  # global - softmax over points
            # PCT: point cloud transformer - regularizing attention at each neighbor location?)
            x = masked_softmax(x, mask, axis=axis, impl='inf') if mask is not None else tf.nn.softmax(x, axis=axis)
        else:
            raise NotImplementedError(f'not supported norm: {n} in {norm_cfg}')

    if mask is not None and n not in ['softmax']:  # mitigate the last mask-out
        x *= mask
    return x
normalize.spatial = ['softmax', 'norm', 'norml1']
normalize.channel = ['l2', 'l1']
normalize.others = ['G_softmax']
normalize.ops = normalize.spatial + normalize.channel + normalize.others

@tf_scope
def apply_kernel(target, kernel, shared_channel=1, reduction='sum', mask=None, axis=-2):
    """
    apply kernel on target: [BxN, k, d_agg] - e.g. perform convolution / weighted aggregation
    """
    # if name:
    #     with tf.variable_scope(name):
    #         return convolute(target, kernel, shared_channel, mask, None)
    mh = shared_channel
    d_k = (target.shape[-1]) // mh
    mask = check_mask(target, mask)  # [BxN, k, 1]

    # reduction as merged normalization (spatial) + sum
    if reduction in normalize.spatial:
        assert kernel is not None
        kernel = normalize(kernel, reduction, mask=mask, axis=axis)
        reduction = 'sum'

    if kernel is None:  # assume kernel applied
        f_out = target
    elif mh > 1:
        assert int(kernel.shape[-1]) == d_k, f'incompatible kernel.shape = {kernel.shape}, but desired d_k = {d_k}'
        target_mh = tf.reshape(target, [-1, tf.shape(target)[-2], d_k, mh])  # [BxN, k,  d_agg // mh, mh]
        kernel_mh = tf.reshape(kernel, [-1, tf.shape(kernel)[-2], d_k, 1])  # [BxN, k/1, d_agg // mh, 1]
        # kernel_mh = tf.expand_dims(kernel, axis=-1)
        f_out = tf.reshape(target_mh * kernel_mh, tf.shape(target))  # [BxN, k, d_agg]
    else:
        f_out = target * kernel

    # mask out jitter
    if mask is not None and kernel is not None:
        f_out *= mask
    # reduction
    f_out = apply_reduction(f_out, reduction, mask, axis=axis)

    return f_out

def apply_reduction(target, reduction, mask, axis=-2):
    """ apply reduction on target
    """
    if reduction == 'avg':
        reduction = 'mean'

    mask = check_mask(target, mask)  # [BxN, k, 1]
    if mask is not None:
        # reduce
        reduce_func = {'sum': tf.reduce_sum, 'mean': tf.reduce_sum, 'max': tf.reduce_max}[reduction]
        f_out = reduce_func(target, axis=axis)  # [BxN, d_agg]

        if reduction == 'mean':  # divide by only valid neighbors
            f_out /= tf.maximum(tf.reduce_sum(mask, axis=axis), _eps)
            # f_out /= (tf.reduce_sum(mask, axis=-2) + _eps)
    else:
        # directly reduce
        f_out = getattr(tf, f'reduce_{reduction}')(target, axis=axis)

    return f_out

def combine(feature_list, ops, mask_list=None, kwargs=None, raise_not_support=True):
    # combine the features (assume has been matched) in the list
    num_f = len(feature_list)
    assert num_f > 1, f'len(feature_list) < 2 for feature_list = {feature_list}'
    rank = len(feature_list[0].shape)
    assert all([rank == len(f.shape) for f in feature_list]), f'unmatched rank for features in list: {feature_list}'
    kwargs = {} if kwargs is None else kwargs
    activation = kwargs.pop('activation') if 'activation' in kwargs else None

    full_shape = None
    if mask_list is not None:
        raise NotImplementedError
        mask_list = mask_list if isinstance(mask_list, list) else [mask_list] * num_f
        full_shape = tf.cast(tf.shape([m for m in mask_list if m is not None][0]), tf.int64)
        full_shape = tf.concat([full_shape, [int(feature_list[0].shape[-1])]], axis=0)  # [BxN (full mask), d (feature)]

    def scatter_feature_list(f_list, m_list, w=None):
        # scatter the feature in f_list back to original shape, according to mask in m_list
        w_list = [w[i] for i in range(len(f_list))] if w is not None else None
        if m_list is not None and w is not None:
            f_list = [f if m is None else tf.scatter_nd(tf.where(m), updates=f*w, shape=full_shape) for w, f, m in zip(w_list, f_list, m_list)]
        elif m_list is not None and w is None:
            f_list = [f if m is None else tf.scatter_nd(tf.where(m), updates=f, shape=full_shape) for f, m in zip(f_list, m_list)]
        elif m_list is None and w is not None:
            f_list = [f*w for w, f in zip(w_list, f_list)]
        return f_list

    if ops.startswith('concat'):
        assert ops in ['concat', 'concatmlp', 'concatlinear']
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.concat(f_list, axis=-1)
        if ops == 'concatmlp':
            features = dense_layer(features, int(f_list[0].shape[-1]), 'concat_mlp', activation, True, True, **kwargs)
        elif ops == 'concatlinear':
            features = dense_layer(features, int(f_list[0].shape[-1]), 'concat_mlp', None, False, False, **kwargs)

    elif ops == 'sum':
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.add_n(f_list)

    elif ops == 'mul':
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.reduce_prod(tf.stack(f_list), axis=0)

    elif ops.startswith('max'):
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.reduce_max(tf.stack(f_list), axis=0)
    
    elif ops.startswith('weight'):
        shape = [num_f, feature_list[0].shape[-1]] if ops.startswith('weights') else [num_f,]
        init = tf.constant(np.ones(shape) * np.log(1 / (shape[0] - 1)), dtype=tf.float32)  # init to mean-pooling (after sigmoid)
        weights = tf_get_variable('weights', shape=None, initializer=init, wd=None)
        weights = tf.reshape(weights, shape[:1] + [1] * (rank - len(shape[1:])) + shape[1:])  # [num_f, 1..., d/1]
        if ops.endswith('Norm'):
            weights = weights / tf.reduce_sum(weights, axis=-2, keepdims=True)
        elif ops.endswith('Softmax'):
            weights = tf.nn.softmax(weights, axis=-2)
        else:  # by default
            weights = tf.nn.sigmoid(weights)  # maps into [0-1]

        if mask_list is None:
            features = tf.concat([tf.expand_dims(f, axis=0) for f in feature_list], axis=0)  # [num_f, ...]
            features = tf.reduce_sum(features * weights, axis=0)
        else:
            f_list = scatter_feature_list(feature_list, mask_list, weights)
            features = tf.add_n(f_list)

    elif ops.startswith('w'):
        # unbound weights (no activation)
        if num_f > 2:
            shape = [num_f, feature_list[0].shape[-1]] if ops.startswith('ws') else [num_f,]
        else:
            shape = [features.shape[-1]] if ops.startswith('ws') else []
        init = ops[2:] if ops.startswith('ws') else ops[1:]
        init = tf.constant(np.ones(shape) * (float(init) if init else 1/num_f), dtype=tf.float32)  # init to mean-pooling, or specified val
        w = tf_get_variable('w', shape=None, initializer=init, wd=None)
        if shape:  # if not scalar
            w = tf.reshape(w, shape[:1] + [1] * (rank - len(shape[1:])) + shape[1:] if num_f > 2 else [1] * (rank - 1) + shape)
        if mask_list is None:
            features = tf.concat([tf.expand_dims(f, axis=0) for f in feature_list], axis=0)  # [num_f, ...]
            features = tf.reduce_sum(features * w, axis=0)
        else:
            f_list = scatter_feature_list(feature_list, mask_list, w=w if num_f > 2 else [w, 1-w])
            features = tf.add_n(f_list)

    elif ops.startswith('gate') and 'Sep' in ops:
        assert mask_list is None
        d_out = feature_list[0].shape[0] if ops.startswith('gates') else 1
        d_weights = d_out 
        w_list = [dense_layer(f, d_weights, f'gate_linear_{i}', None, True, False, **kwargs) for i, f in enumerate(feature_list)]

        if len(ops.split('Sep')) > 1:  # gate[s]Sep.*
            features = tf.concat([tf.expand_dims(f, axis=-2) for f in feature_list], axis=-2)  # [BxN, num_f, d]
            weights = tf.concat([tf.expand_dims(w, axis=-2) for w in w_list], axis=-2)  # [BxN, num_f, d/1]
            weights = normalize(weights, ops.split('Sep')[-1]) if ops.split('Sep')[-1] else weights
            features = tf.reduce_sum(features * weights, axis=-2)
        else:
            features = tf.add_n(f*w for f, w in zip(w_list, feature_list))

    elif ops.startswith('gate'):
        assert mask_list is None
        d_in = int(feature_list[0].shape[-1])  # d = d_in
        shape = tf.shape(feature_list[0])  # [..., d]
        d_out = int(shape[-1]) if ops.startswith('gates') else 1  # sigmoid - d/1 - if modulating feature dim, or one weight each feature vec)
        d_out = d_out if num_f == 2 else d_out * num_f  # norm (e.g. softmax) - 1/num_f * d/1 - if need to normalize (num_f > 2), or using (s, 1-s)
        features = tf.concat(feature_list, axis=-1)  # [BxN, num_f x d]
        weights = dense_layer(features, d_out, 'gate_linear', None, True, False, **kwargs)
        if num_f == 2:  # sigmoid
            weights = tf.nn.sigmoid(weights)  # [..., d/1]
            features = feature_list[0] * weights + feature_list[1] * (1 - weights)
        else:  # norm - norm or softmax
            shape_w = tf.concat([shape[:-1], [num_f, int(d_out / num_f)]], axis=0)  # [..., num_f, d/1]
            shape_f = tf.concat([shape[:-1], [num_f, d_in]], axis=0) if int(d_out / num_f) != d_in else shape_w   # [..., num_f, d]
            assert int(d_out / num_f) in [1, d_in], f'wrong calc on shape_w = [..., {num_f}, {int(d_out / num_f)}], but feature = {feature_list[0]}'
            weights = tf.reshape(weights, shape_w)
            weights = normalize(weights, ops.split('-')[-1]) if '-' in ops else tf.nn.softmax(weights, axis=-2)
            features = tf.reshape(features, shape_f)
            print(features, weights, features * weights)
            features = tf.reduce_sum(features * weights, axis=-2)
        if 'mlp' in ops.split('-')[-1]:
            features = dense_layer(features, int(shape[-1]), 'gate_mlp', activation, True, True, **kwargs)

    elif raise_not_support:
        raise NotImplementedError(f'not supported ops = {ops}')
    else:
        features = None
    return features

@tf_scope
def drop_path(features, rate, is_training, scale_by_keep=True, batch_inds=None):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        - drops out a whole example hiddenstate with the specified probability
    Args:
        rage: float - rate to drop, ie. survival rate = 1 - rate
    """
    keep_prob = 1 - rate

    def _drop(x):
        # gen rand
        B = batch_inds[-1] + 1 if batch_inds is not None else tf.shape(x)[0]
        rand = tf.random.uniform(shape=[B] + [1] * (x._rank() - 1), maxval=1) + keep_prob
        rand = tf.math.floor(rand)  # [B, 1...] - binarized
        if scale_by_keep:
            rand /= keep_prob
        if batch_inds is not None:
            rand = tf.gather(rand, batch_inds)
        # rand - [BxN/B, 1...]
        return x * rand

    features = tf_cond(is_training, true_fn=lambda: _drop(features), false_fn=lambda: features)
    return features

@tf_scope
def dropout(features, rate, is_training, noise_shape=None):
    """ Dropout layer.
    Args:
        noise_shape: None / list of ints
    """
    features = tf_cond(is_training,
        true_fn=lambda: tf.nn.dropout(features, rate=rate, noise_shape=noise_shape),
        false_fn=lambda: features,
    )
    return features

def prelu(x, init=None, mh=1, shape=None):
    # parametric relu, following pytorch implementation, except for init value - 0 instead of 0.25

    if shape is not None:
        raise NotImplementedError(f'not supported to specify shape = {shape}')  # allow per-input alpha?
    else:
        assert mh == 1, f'not supported vector alpha'
        shape = []

    alpha = tf_get_variable('alpha', shape, initializer='zeros', wd=None)
    if isinstance(init, float):
        alpha += init

    if mh > 1:  # vector - match dims
        shape = [1] * (len(x.shape) - 1) + shape
        alpha = tf.reshape(x, shape)

    pos = tf.nn.relu(x)
    neg = -alpha * tf.nn.relu(-x)
    return pos + neg

def gelu(x, approximate=False):
    """ Gaussian error linear unit (GELU)
    weights inputs by their value, rather than gates inputs by their sign as in ReLU:
    computes x * P(X <= x), where P(X) ~ N(0, 1)
    """
    assert x.dtype in [tf.float32]
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        x = 0.5 * x * (1.0 + tf.math.tanh(tf.cast(0.7978845608028654, x.dtype) * (x + coeff * x ** 3)))
    else:
        x = 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))
    return x
