import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

import re
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf
fetch_supports = fetch_supports_flow

# ---------------------------------------------------------------------------- #
# block ops
# ---------------------------------------------------------------------------- #

def unary_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    """
    Block performing a simple 1x1 convolution (shared mlp)
    """
    act = None if block_cfg.linear else config.activation
    bn = not block_cfg.linear

    x = dense_layer(features, d_out, None, act, True, bn, is_training, config.init,
                    weight_decay=config.weight_decay, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)
    return x

def agg_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training, support_idx=None):
    """
    Local ops to aggregate features from neighborhood
    """

    pts, neighbor_idx = fetch_supports(inputs, stage_n, stage_i)  # [BxN, k]
    neighbor_idx = support_idx if support_idx is not None else neighbor_idx
    batch_dims = len(neighbor_idx.shape) - 2
    has_self_loop = config.sample not in ['grid'] or not stage_n  # not on up/down sampling, or not using grid sampling

    reduction = block_cfg.reduce
    if reduction in ['dist']:
        # if has_self_loop:  # mask out 0-dist (self-loop)
        #     neighbor_idx = neighbor_idx[..., 1:] if not batch_dims else neighbor_idx[..., 1:]
        if block_cfg.k:
            neighbor_idx = neighbor_idx[..., :block_cfg.k]  # restrict max neighbors
        center_xyz = inputs['points'][stage_i]  # [BxN, 3]
        neighbor_xyz = tf_gather(pts, neighbor_idx, get_mask=False)  # [BxN, k, 3]

    kernel = None
    shadow_fn = 'min' if reduction == 'max' else tf.zeros
    neighbor_features, mask = tf_gather(features, neighbor_idx, shadow_fn=shadow_fn)
    target = {'fj': neighbor_features}[block_cfg.agg]  # [BxN, k, d_agg]
    shared_channel = 1

    # apply ops
    if reduction == 'dist':  # inverse distance weighting
        kr = get_kr(config, stage_n, stage_i)
        kernel = position_encode(center_xyz, neighbor_xyz, norm=kr, encoding=block_cfg.enc)  # [BxN, k, 1]
        kernel = 1 / (kernel + _eps)
        kernel = normalize(kernel, block_cfg.norm, mask=mask)
        reduction = 'sum'
        mh = int(kernel.shape[-1])
        if mh > 1:  # dividing feature into groups
            d_out = int(features.shape[-1])
            assert d_out // mh == 0, f'fdims = {d_out} not compatible with mh = {mh}'
            shared_channel = int(d_out / mh)

    output = apply_kernel(target, kernel, shared_channel=shared_channel, reduction=reduction, mask=mask)

    # if has_self_loop and mask is not None:  # add back self-loop (point with no neighbors)
    #     single_mask = tf.cast(tf.reduce_sum(mask, axis=-1, keepdims=True) < 1, tf.float32)
    #     output += features * single_mask

    return output

def sample_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    """
    various sampling scheme
    """
    _, neighbor_idx = fetch_supports(inputs, stage_n, stage_i)  # [BxN, k]

    if block_cfg.sample == 'nst':  # nearest sampling
        batch_dims = 0 if 'batches_len' in inputs else 1
        features = tf.gather(features, neighbor_idx[..., 0], batch_dims=batch_dims)  # [BxN, d]
    else:
        # e.g. local aggregation as sampling - dist, mean, max, sum
        features = apply_block_ops(features, d_out, inputs, stage_n, stage_i, block_cfg.ops, config, is_training)

    return features

def upsample_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    """
    upsampling + join features from downsampling stage
    """
    assert stage_n == 'up'
    kwargs = get_kwargs(block_cfg, config, is_training)
    mlp_kwargs = get_kwargs_mlp(block_cfg, config, is_training)

    feat_skip = inputs['stage_list']['down'][stage_i]['f_out']  # skip connection
    if block_cfg.squeeze and block_cfg.squeeze > 1:  # squeeze the d_out after join
        d_out /= block_cfg.squeeze
    d_mid = feat_skip.shape[-1]  # matching the down stage (from skip connection)

    # transform-pre
    if block_cfg.f:  # transform on features
        features = mlps_by_ops(features, d_mid, d_mid, block_cfg.f, mlp_kwargs, name='feat')
    if block_cfg.s:  # transform on skip connection
        feat_skip = mlps_by_ops(feat_skip, d_mid, d_mid, block_cfg.s, mlp_kwargs, name='skip')

    # sample
    features = sample_block(features, d_mid, inputs, stage_n, stage_i, block_cfg, config, is_training)

    # join
    for ops in block_cfg.join.split('-'):
        if ops == 'concat':
            features = tf.concat([features, feat_skip], axis=-1)
        elif ops == 'sum':
            features += feat_skip
        elif 'mlp' in ops or ops == 'linear':
            features = mlps_by_ops(features, d_out, d_out, ops, mlp_kwargs)
        else:
            raise NotImplementedError(f'not implemented ops = {ops} in join type = {join}')

    # transform-post
    if int(features.shape[-1]) != d_out:
        # features = mlps(features, d_out, d_out, block_cfg.out, linear=block_cfg.out == 'linear', name='out', **mlp_kwargs)
        features = dense_layer(features, d_out, 'mlp_out', config.activation, True, True, **kwargs)
    return features

def conv_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training, kernel_f=None, support_idx=None):
    """
    convolution - kenerl genenrated from mlp(s)
    """
    kwargs = get_kwargs(config, config, is_training)  # backbone setting
    kwargs_mlp = get_kwargs_mlp(config, config, is_training)

    d_in = int(features.shape[-1])
    pts, neighbor_idx = fetch_supports(inputs, stage_n, stage_i)  # [BxN, k]
    neighbor_idx = support_idx if support_idx is not None else neighbor_idx

    center_xyz = inputs['points'][stage_i]  # [BxN, 3]
    neighbor_xyz, mask = tf_gather(pts, neighbor_idx, get_mask=True)  # [BxN, k, 3]
    neighbor_features = tf_gather(features, neighbor_idx, get_mask=False)  # [BxN, k, d]

    # kernel encoding
    if kernel_f is None:
        norm = get_kr(config, stage_n, stage_i)
        kernel_f = relative_encoding(block_cfg.enc, features, neighbor_features, center_xyz, neighbor_xyz, norm, d_in, kwargs_mlp)

    # generate kernel
    mh = int(block_cfg.mh) if block_cfg.mh else 1
    with tf.variable_scope('kernel'):
        kernel_f = tf.concat(kernel_f, axis=-1) if len(kernel_f) > 0 else kernel_f[0]  # [BxN, k, d_f]
        target = {'fj': neighbor_features, 'kernel': kernel_f}[block_cfg.agg]  # [BxN, k, d_agg]
        d_k = int(target.shape[-1]) // mh
        d_k_mid = block_cfg.kerneldim if block_cfg.kerneldim else min(config.first_features_dim, d_k)

        # kernel generation - mlp setting from backbone
        kernel_kwargs = get_kwargs_mlp(block_cfg.kernel_cfg, config, is_training, bn=False)  # conv kernel mlp
        kernel = mlps_by_ops(kernel_f, d_k_mid, d_k, block_cfg.kernel, kwargs=kernel_kwargs)  # [BxN, k, d_agg // mh]
        with tf.variable_scope('norm'):
            if block_cfg.norm in normalize.ops:
                kernel = normalize(kernel, block_cfg.norm, mask=mask)
            elif block_cfg.norm == 'softmaxR':  # softmax-rescale
                rescale = int(kernel.shape[-1]) ** -0.5
                kernel = normalize(kernel, 'softmax', mask=mask) * rescale
            elif block_cfg.norm:
                raise ValueError(f'not support norm = {block_cfg.norm}')
            else:
                pass

        if block_cfg.convwd:  # add weight_decay on generated kernel
            conv_weight_decay = tf.multiply(tf.nn.l2_loss(kernel), block_cfg.convwd, name='weight_loss')
            tf.add_to_collection('weight_losses', conv_weight_decay)

    # mask = tf.Print(mask, [tf.shape(mask), tf.reduce_max(neighbor_idx), tf.reduce_max(tf.reduce_sum(mask, axis=-1)), '[[]]', tf.reduce_sum(mask, axis=-1)], summarize = 1000)
    f_out = apply_kernel(target, kernel, shared_channel=mh, reduction=block_cfg.reduce, mask=mask, name='convolute')  # [BxN, d_agg]

    # post-conv - by default: bn-act
    for ops in block_cfg.proj.split('-'):
        if ops == 'bn':  # normalization
            if block_cfg.bn == 'bn':
                f_out = tf.layers.batch_normalization(f_out, momentum=config.bn_momentum, epsilon=config.bn_eps, training=is_training, fused=False)
            elif block_cfg.bn:
                raise NotImplementedError(f'not supported bn = {block_cfg.bn} in conv')
        elif ops == 'act':  # activation
            act = block_cfg.act if block_cfg.act else config.activation
            f_out = get_activation(act)(f_out)
        else:  # projection mlps (if any)
            f_out = mlps_by_ops(f_out, d_out, d_out, ops, name='mlp_out', kwargs=kwargs_mlp)

    if int(f_out.shape[-1]) != d_out:  # potential out mlp to match dims
        f_out = dense_layer(f_out, d_out, 'mlp_out', config.activation, True, True, **kwargs)
    return f_out

def lfa_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    """
    local feature aggregation - RandLA-Net
    """
    kwargs = get_kwargs(config, config, is_training)  # backbone setting
    kwargs_mlp = get_kwargs_mlp(block_cfg, config, is_training)

    d_in = int(features.shape[-1])
    pts, neighbor_idx = fetch_supports(inputs, stage_n, stage_i)  # [BxN, k]

    center_xyz = inputs['points'][stage_i]  # [BxN, 3]
    neighbor_xyz, mask = tf_gather(pts, neighbor_idx, get_mask=True)  # [BxN, k, 3]
    neighbor_features = tf_gather(features, neighbor_idx, get_mask=False)  # [BxN, k, d]

    # kernel encoding
    norm = get_kr(config, stage_n, stage_i)
    kernel_f = relative_encoding(block_cfg.enc, features, neighbor_features, center_xyz, neighbor_xyz, norm, d_in, kwargs_mlp)

    repeat = int(block_cfg.repeat)
    for r in range(repeat):
        d_in = int(features.shape[-1])
        d_mid = d_out * int(block_cfg.expand) ** r  # ratio of expanding the feature dims while stacking ops
        with tf.variable_scope(f'agg_{r}'):
            # => no transform in `relative_encoding` - done here
            kernel_f[0] = mlps_by_ops(kernel_f[0], d_in, d_in, ops=block_cfg.encproj, kwargs=kwargs_mlp, name='pos')  # using last f_xyz
            kernel_f[1] = tf_gather(features, neighbor_idx, get_mask=False) if r > 0 else neighbor_features  # using last f
            features = conv_block(features, d_mid, inputs, stage_n, stage_i, block_cfg, config, is_training, kernel_f=kernel_f)
    return features

def attention_block(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    """
    Q-K-V attention
    """
    f_in = features
    d_in = int(features.shape[-1])
    d_mid = d_in
    kwargs = get_kwargs(block_cfg, config, is_training=is_training)
    mlp_kwargs = get_kwargs_mlp(block_cfg, config, is_training=is_training)

    activation = mlp_kwargs['activation']
    if block_cfg.norm in ['pre', 'post']:
        norm_ops = mlp_kwargs['bn']
        norm_proj = mlp_kwargs['bn'] = kwargs['bn'] = ''
    else:
        assert block_cfg.norm == ''
        norm_proj = mlp_kwargs['bn'] if 'bn' in mlp_kwargs else True

    BN_shape = [-1] if 'batches_len' in inputs else [tf.shape(features)[0], features.get_shape().as_list()[1]]
    BN_shape = [i if i is not None else -1 for i in BN_shape]
    has_self_loop = config.sample not in ['grid'] or not stage_n  # not on up/down sampling, or not using grid sampling - neighbor[0] = pts = center

    pts, neighbor_idx = fetch_supports(inputs, stage_n, stage_i)  # [BxN, k]
    center_xyz = inputs['points'][stage_i]  # [BxN, 3]
    neighbor_xyz, mask = tf_gather(pts, neighbor_idx, get_mask=True)  # [BxN, k, 3]

    # prepare attention input
    if block_cfg.ratio:
        d_mid = d_in // int(block_cfg.ratio)
        features = dense_layer(features, d_mid, 'linear_in', None, True, norm_proj, **kwargs)  # linear projection - in
    if block_cfg.norm == 'pre':
        features = norm(features, norm_ops)
    features_k = tf_gather(features, neighbor_idx, get_mask=False)  # [BxN, k, d]
    if stage_n:  # generate primitive feat for Q
        features = apply_block_ops(features, d_mid, inputs, stage_n, stage_i, block_cfg.pool, config, is_training)
        if block_cfg.norm == 'pre':
            features = norm(features, norm_ops)

    # prepare position encoding (shared)
    with tf.variable_scope('position_encoding'):
        pos_enc = None
        xyz_list = []  # collect required position
        if 'pos' in block_cfg.q and not has_self_loop:  # center - add self loop as required
            xyz_list += [tf.expand_dims(center_xyz, axis=-2)]  # [BxN, 1, 3]
        if any(['pos' in ops for ops in [block_cfg.q, block_cfg.k, block_cfg.v, block_cfg.A]]):  # neighbor
            xyz_list += [neighbor_xyz]
        if xyz_list:  # if using position encoding
            pos_enc = tf.concat(xyz_list, axis=-2) if len(xyz_list) > 1 else xyz_list[0]
            # shared pos_enc transform
            kr = get_kr(config, stage_n, stage_i)
            pos_enc = position_encoding_transform(ops=block_cfg.pos, d_out=d_mid, center_xyz=center_xyz, neighbor_xyz=pos_enc, norm=kr, activation=activation, **kwargs)
            # # separate pos_enc transform for each branch
            # pos_enc = position_encode(center_xyz, pos_enc, norm=kr, encoding='-'.join(pos_list))
            pos_enc_q = pos_enc[..., 0, :]  # [BxN, d_mid]
            pos_enc_k = pos_enc[..., 1:, :] if len(xyz_list) > 1 else pos_enc  # [BxN, k, d_mid]

    @tf_scope
    def join_position_encoding(features, pos_enc, ops):
        if len(features.shape) < len(pos_enc.shape):  # gather to match pos_enc
            features = tf_gather(features, neighbor_idx, get_mask=False)  # [BxN, k, d]
        # # transform the position encoding (sharing controlled by scope???)
        # f_xyz = position_encoding_transform(ops=block_cfg.pos, d_out=d_mid, pos_enc=pos_enc, activation=activation, **kwargs)
        f_xyz = pos_enc  # shared transform

        join = ops[len('pos'):]
        if join == 'C':
            features = tf.concat([features, f_xyz], axis=-1)
        elif join == '':  # default to sum
            features += f_xyz
        else:
            raise NotImplementedError(f'not supported join type for position encoding {join} from {ops}')
        return features

    @tf_scope
    def norm(f, ops):
        if ops == 'bn':
            f = tf.layers.batch_normalization(f, momentum=kwargs['bn_momentum'], epsilon=kwargs['bn_eps'], training=is_training, fused=False)
        elif ops == 'ln':
            b_inds = get_batch_inds(inputs, stage_i) if 'batches_len' in inputs else None
            b_len = inputs['batches_len'][stage_i] if 'batches_len' in inputs else None
            f = group_norm(f, groups=1, group_size=None, gamma=None, beta=None, eps=kwargs['bn_eps'], batches_len=b_len, batches_ind=b_inds)
        else:
            raise ValueError(f'not support norm={ops}')
        return f

    @tf_scope
    def sequential_transform(features, pos_enc, ops_str, fdims=None):
        fdims = fdims if fdims is not None else d_mid
        for ops in [i for i in ops_str.split('-')]:
            if ops.startswith('pos'):
                features = join_position_encoding(features, pos_enc, ops, name='pos')
            elif 'mlp' in ops or 'lin' in ops:
                features = mlps_by_ops(features, fdims, fdims, ops=ops, kwargs=mlp_kwargs)
            else:
                raise NotImplementedError(f'not supported transform {ops} in {ops_str}')

        if len(features.shape) == len(neighbor_idx.shape):
            features = tf.expand_dims(features, axis=-2)
        return features

    @tf_scope
    def apply_drop_path(x, rate, is_training):
        b_inds = None
        if 'batches_len' in inputs:
            b_inds = get_batch_inds(inputs, stage_i)  # [BxN] - perpoint batch inds
        x = drop_path(x, float(rate), is_training, batch_inds=b_inds)
        return x

    # per-branch process
    # branch sharing via scope re-use ?    
    scope_kwargs = {'K': {'name': 'K'}, 'V': {'name': 'V'}}
    if 'q' in block_cfg.share:
        if 'k' in block_cfg.share:
            scope_kwargs['K']['reuse'] = 'Q'
        if 'v' in block_cfg.share:
            scope_kwargs['V']['reuse'] = 'Q'
    elif 'k' in block_cfg.share:
        if 'v' in block_cfg.share:
            scope_kwargs['V']['reuse'] = 'K'

    # apply transform
    Q = sequential_transform(features, pos_enc_q, block_cfg.q, name='Q')  # [BxN, d]
    K = sequential_transform(features_k, pos_enc_k, block_cfg.k, **scope_kwargs['K'])  # [BxN, k, d]
    V = sequential_transform(features_k, pos_enc_k, block_cfg.v, **scope_kwargs['V'])

    mh = d_mid if block_cfg.mh == 'd' else int(block_cfg.mh)  # multi-head defined by shared channel (vec length of in a head) -  #head
    pos_enc_A = pos_enc_k
    head_num = mh
    d_h = d_mid // head_num
    assert d_h > 0, f'incompatible mh - d_mid={d_mid} and head_num={head_num} give d_h={d_h}'

    neighbor_k = neighbor_idx.get_shape().as_list()[-1]
    if neighbor_k is None:
        neighbor_k = tf.shape(neighbor_idx)[-1]
    Q = tf.reshape(Q, [-1, 1, head_num, d_h])
    K = tf.reshape(K, [-1, neighbor_k, head_num, d_h])
    V = tf.reshape(V, [-1, neighbor_k, head_num, d_h])
    pos_enc_A = tf.reshape(pos_enc_A, [-1, neighbor_k, head_num, d_h])
    if mask is not None:
        mask = tf.reshape(mask, [-1, neighbor_k, 1, 1])  # [BxN, k, 1, 1]

    # Q-K => generate attention mask
    A = None
    for ops in block_cfg.A.split('-'):
        ops_name = re.match('[a-z]*', ops).group()
        with tf.variable_scope(f'A/{ops_name}'):
            if ops == 'cos':
                Q = tf.nn.l2_normalize(Q, axis=-1, epsilon=_eps)
                K = tf.nn.l2_normalize(K, axis=-1, epsilon=_eps)
            # Q-K
            if ops in ['dot', 'cos']:
                A = tf.reduce_sum(Q * K, axis=-1, keepdims=True)
            elif ops in ['add']:
                A = K + Q
            elif ops in ['minus']:
                A = K - Q
            elif ops.startswith('mh') and 'mlp' in ops or 'linear' in ops:  # mlp across head
                A = tf.reshape(A, [-1, int(neighbor_idx.shape[-1]), d_mid])
                A = sequential_transform(A, None, ops)
                A = tf.reshape(A, [-1, int(neighbor_idx.shape[-1]), head_num, d_h])
            elif 'mlp' in ops or 'linear' in ops:  # per-head mlp
                A = sequential_transform(A, None, ops, fdims=d_h)
            elif ops.startswith('pos'):
                A = join_position_encoding(A, pos_enc_A, ops)
            # normalize
            elif ops in ['rescale', 'scale']:
                A /= tf.sqrt(d_h)
            elif ops.startswith('drop'):
                A = tf.cond(is_training, lambda: tf.nn.dropout(A, rate=float(ops[len('drop'):])), lambda: A)
            elif ops in normalize.spatial:
                A = normalize(A, ops, mask=mask, axis=-3)
            elif ops in normalize.channel:
                A = normalize(A, ops, mask=mask)
            else:
                raise NotImplementedError(f'not supported ops for A: {ops} in {block_cfg.A}')
        # print('A\t', ops_name, A)
    reduction = block_cfg.reduce
    with tf.variable_scope('A'):
        if block_cfg.scale:
            A /= tf.math.sqrt(float(d_h))
        if reduction in normalize.spatial:
            A = normalize(A, reduction, mask=mask, axis=-3)
            reduction = 'sum'
        if block_cfg.drop_att:
            A = dropout(A, float(block_cfg.drop_att), is_training, name='drop')

    # apply attention on V
    f_att = apply_kernel(V, A, reduction=reduction, mask=mask, axis=-3, name='convolute')  # [BxN, k, #head, d_h]
    f_att = tf.reshape(f_att, BN_shape + [d_mid])  # [BxN, d_agg] - from [BxN, #head, d_h]

    f_att = dense_layer(f_att, d_out, 'projection', None, True, norm_proj, **kwargs)  # linear projection - out
    with tf.variable_scope('shortcut'):
        if int(f_in.shape[-1]) != d_out:
            f_in = dense_layer(f_in, d_out, 'proj_in', None, True, norm_proj, **kwargs)
        if block_cfg.drop:
            f_att = dropout(f_att, float(block_cfg.drop), is_training, name='drop')
        if block_cfg.drop_path:
            f_att = apply_drop_path(f_att, float(block_cfg.drop_path), is_training, name='drop_path')
        f_att += f_in

    if block_cfg.norm == 'post':
        f_att = norm(f_att, norm_ops)

    with tf.variable_scope('ffn'):
        if block_cfg.ffn:
            f_ffn = f_att
            if block_cfg.norm == 'pre':
                f_ffn = norm(f_ffn, norm_ops)
            d_mid = int(d_out * float(block_cfg.ffn_ratio)) if block_cfg.ffn_ratio else d_out
            f_ffn = mlps_by_ops(f_ffn, d_mid, d_out, ops=block_cfg.ffn, kwargs=mlp_kwargs)
            if block_cfg.drop:
                f_ffn = dropout(f_ffn, float(block_cfg.drop), is_training, name='drop')
            if block_cfg.drop_path:
                f_ffn = apply_drop_path(f_ffn, float(block_cfg.drop_path), is_training, name='drop_path')
            f_att += f_ffn
            if block_cfg.norm == 'post':
                f_att = norm(f_att, norm_ops)

    return f_att

def bottleneck(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training, **kwargs):
    """
    Block performing a resnet bottleneck, potentially strided for sampling stage
    if used in down/up sampling stage:
        spatial sampling - fetch the sampled pc > process
        dynamic sampling - generate the sampled pc > process > update inputs
    """

    d_in = int(features.shape[-1])
    d_mid = int(d_out / block_cfg.ratio)
    depth = int(block_cfg.depth)

    initializer = config.init
    activation = config.activation
    weight_decay = config.weight_decay
    bn_momentum = config.bn_momentum
    bn_eps = config.bn_eps

    with tf.variable_scope('conv1'):
        x = dense_layer(features, d_mid, None, activation, True, True, is_training, initializer,
                        weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps)

    with tf.variable_scope('conv2'):  # invoke block ops (striding handled by the ops)
        with tf.variable_scope(block_cfg.ops.name):
            x = apply_block_ops(x, d_mid, inputs, stage_n, stage_i, block_cfg.ops, config, is_training, **kwargs)
        for i in range(1, depth):
            with tf.variable_scope(f'{block_cfg.ops.name}_{i}'):
                x = apply_block_ops(x, d_mid, inputs, stage_n, stage_i, block_cfg.ops, config, is_training, **kwargs)

    with tf.variable_scope('conv3'):
        x = dense_layer(x, d_out, None, None, True, True, is_training, initializer,
                        weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps)

    with tf.variable_scope('shortcut'):
        shortcut = features
        if stage_n:  # if is sampling stage
            # non-param ops would ignore 'd_out', e.g. max/avg/nst/int-pooling
            with tf.variable_scope(block_cfg.pool.name):
                shortcut = apply_block_ops(shortcut, d_out, inputs, stage_n, stage_i, block_cfg.pool, config, is_training, **kwargs)

        if int(shortcut.shape[-1]) != d_out:  # plain - linear + bn
            shortcut = dense_layer(shortcut, d_out, 'conv_1x1', None, True, True, is_training, initializer,
                            weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps)

    x += shortcut
    output = get_activation(activation)(x)
    return output

def get_block_ops(block_n, raise_not_found=True):

    # resnet bottleneck w/o strided
    if block_n.startswith('resnetb'):
        block_ops = bottleneck

    # mlps
    elif block_n in ['unary', 'linear']:
        block_ops = unary_block

    # simple aggregation
    elif block_n.startswith('agg') or block_n.startswith('pool') or block_n in ['distconv']:
        block_ops = agg_block

    # sampling
    elif 'sample' in block_n:
        block_ops = globals()[f'{block_n}_block']

    # lfa
    elif block_n == 'lfa':
        block_ops = lfa_block

    elif block_n.startswith('attention'):
        block_ops = attention_block

    # raise or skip
    elif raise_not_found:
        raise NotImplementedError(f'not supported block_n = {block_n}')
    else:
        block_ops = None
    return block_ops

@tf_scope
def apply_block_ops(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training):
    block_ops = get_block_ops(block_cfg.name)
    features = block_ops(features, d_out, inputs, stage_n, stage_i, block_cfg, config, is_training)
    return features
