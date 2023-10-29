import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

import re
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf
from .blocks import apply_block_ops
fetch_supports = fetch_supports_stage

from collections import defaultdict

# ---------------------------------------------------------------------------- #
# heavy & re-uesable func
# ---------------------------------------------------------------------------- #

def get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='max', extend=False):
    """ collect label for sub-sampled points via accumulated sampling ratio
    """
    scene_neighbor = get_sample_idx(inputs, 'U0', (stage_n, stage_i), ftype, config)  # NOTE: must have ftype at U0
    if scene_neighbor is None and extend:  # extend first-layer (no sub-sampling) with neighbor_idx
        scene_neighbor = inputs['neighbors'][stage_i]
    if scene_neighbor is None:
        return None

    _glb = inputs['_glb']
    ftype, ptype = get_ftype(ftype)
    key = f'{stage_n}{stage_i}/{ptype}/scene_label-{reduction}'
    if key not in _glb:
        # gather labels
        key_gather = f'{stage_n}{stage_i}/{ptype}/scene_label-gather'
        if key_gather not in _glb:
            labels = tf_gather(inputs['point_labels'], scene_neighbor, shadow_fn=-1, get_mask=False)  # [BxN, k] - invalid label (-1) one_hot to 0s
            valid_mask = tf.greater_equal(labels, 0)
            labels = tf.one_hot(labels, depth=config.num_classes, axis=-1)
            _glb[key_gather] = (labels, valid_mask)
        labels, valid_mask = _glb[key_gather]
        # summarize labels
        labels = get_neighbor_summary(tf.cast(labels, tf.float32), valid_mask=valid_mask, reduction=reduction)
        _glb[key] = labels
    return _glb[key]  # [BxN, 1/num_classes]


def get_sample_idx(inputs, stage_from, stage_to, ftype, config, kr=None):
    """collect neighbors from up/sub-sampled points, by accumulated sub-sampling ratio
        => neighbor_idx indexing stage_from pts - [stage_to pts, num of neighbor in stage_from pts]
        i.e. stage_from = support, stage_to = queries
    """
    _, ptype = get_ftype(ftype)
    n_from, i_from = parse_stage(stage_from, config.num_layers)[0] if isinstance(stage_from, str) else stage_from
    n_to, i_to = parse_stage(stage_to, config.num_layers)[0] if isinstance(stage_to, str) else stage_to

    pts_from = inputs['stage_list'][n_from][i_from][ptype]
    pts_to = inputs['stage_list'][n_to][i_to][ptype]

    if i_from - i_to == 0:  # no sample
        assert pts_from == pts_to, f'pts have changed from {stage_from} to {stage_to} ({ftype})'
        return None

    # upsampling: i_to closer to end - up0, compared with e.g. down4/up4
    # downsampling: i_to closer to last - down4, compared with e.g. down0
    updown = 'up' if i_to < i_from else 'down'
    # assert n_to == 'up' or n_from == n_to, f'not supported sampling from {stage_from} to {stage_to}'

    if abs(i_from - i_to) == 1:  # sample to next/last stage
        neighbor_idx = inputs['sample_idx'][updown][i_from]
        return neighbor_idx[..., 0] if kr == 1 else neighbor_idx[...,:kr] if isinstance(kr, int) else neighbor_idx

    # down/up-sample more than 1 stage
    _glb = inputs['_glb']
    key = f'{n_from}{i_from}-{n_to}{i_to}/{ftype}/sample_neighbor'
    if key not in _glb:

        from ops import get_tf_func
        search_func = get_tf_func(config.search)
        # search kr depending on (cumulated) radius/ratio in sub-sampling points

        # grid - down: r_sample[i_to - 1]; up: r_sample[i_from - 1]  => using the larger one
        # knn - donw: r_sample[i_from:i_to]; up: r_sample[i_to:i_from]  => smaller to larger
        i_min, i_max = min(i_from, i_to), max(i_from, i_to)

        kr_search = config.r_sample[i_max - 1] if config.sample == 'grid' else kr if kr else np.prod(config.r_sample[i_min:i_max])
        args = [inputs['batches_len'][i_to], inputs['batches_len'][i_from], kr_search] if config.sample == 'grid' else [kr_search]
        neighbor_idx = search_func(pts_to, pts_from, *args, device='/cpu:0')  # queries, supports, ...

        _glb[key] = neighbor_idx[..., 0] if kr == 1 else neighbor_idx
    return _glb[key]

@tf_scope
def batch_reduce(features, inputs, stage_i, config, func, shadow=tf.zeros):
    if isinstance(func, str):
        func = getattr(tf, func)

    if 'batches_len' not in inputs:
        # features - [B, N, ...]
        return func(features, axis=1)

    # features - [BxN, ...]
    assert config.search == 'radius'
    batch_mask = stack_batch_inds(inputs, stage_i=stage_i)  # [B, N_max = max points of point cloud]
    features = tf_gather(features, batch_mask, get_mask=False, shadow_fn=shadow)  # [B, N_max, ...]
    features = func(features, axis=1)
    return features  # [B, ...]

@tf_scope
def calc_dist(f_from, f_to, dist, align=True, keepdims=True):
    # f_from / f_to - prepared & normalized in advance
    assert dist in calc_dist.valid, f'invalid dist={dist}'
    if dist in ['l2', 'norml2']:
        dist = tf.reduce_sum((f_from - f_to)**2, axis=-1, keepdims=keepdims)
        dist = tf.sqrt(tf.maximum(dist, _eps))  # avoid 0-distance - nan due to tf.sqrt numerical unstable at close-0
        # dist = tf.where(tf.greater(dist, 0.0), tf.sqrt(dist), dist)
    elif dist in ['l2square', 'norml2square']:
        dist = tf.reduce_sum((f_from - f_to)**2, axis=-1, keepdims=keepdims)
    elif dist in ['l1', 'norml1']:
        dist = tf.reduce_sum(tf.abs(f_from - f_to), axis=-1, keepdims=keepdims)
    elif dist in ['dot', 'normdot']:  # normdot = raw cos
        # perm = list(range(len(f_to.shape)))
        # f_to = tf.transpose(f_to, perm=perm[:-2] + perm[:-2][::-1])
        # dist = tf.matmul(f_from, f_to)
        dist = tf.reduce_sum(f_from * f_to, axis=-1, keepdims=keepdims)
        dist = -dist if align else dist  # revert to (-inf i.e. small <- similar, dissim -> inf i.e. large)
    elif dist == 'cos':
        dist = tf.reduce_sum(f_from * f_to, axis=-1, keepdims=keepdims)
        # NOTE: matmul seems incorrectly used - training diverge
        # dist = tf.matmul(f_to, f_from, transpose_b=True)  # [k, 1] = [k, d] @ [1, d]^T - similar = smaller
        dist = -dist if align else dist  # revert to (-1 <- similar, dis-similar -> 1)
        dist = (1 + dist) / 2  # rescale to (0, 1)
    elif dist == 'kl':
        # f_from/to need to be a distribution - (0 <- sim, dis -> inf)
        dist = tf.reduce_sum(tf.math.xlogy(f_from, f_from / tf.maximum(f_to, _eps)), axis=-1, keepdims=keepdims)
    else:
        raise NotImplementedError(f'not supported dist = {dist}')
    return dist  # [BxN, k, 1] / [BxN, 1]
calc_dist.valid = ['cos', 'l2', 'norml2', 'l1', 'norml1', 'dot', 'normdot', 'l2square', 'norml2square', 'kl']

def get_class_weight(dataset, labels=None, weight_type='class', normalize='', version=''):
    class_cnt = {
        'S3DIS': np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                            650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32),

        'Semantic3D': np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353], dtype=np.int32),

        'SemanticKITTI': np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                    240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                    9833174, 129609852, 4506626, 1168181], dtype=np.int32),

        'ScanNet': np.array([13665261, 12917093, 2012834, 1413473, 3505649, 1335056, 1996113, 2054593,
                                1425823, 1062475, 235885, 264133, 976177, 891644, 258407, 136529, 149729,
                                120299, 175243, 1693931], dtype=np.int32),

        'SensatUrban': np.array([34658128, 46204961, 70354513, 2034843, 238706, 3805852, 42245,
                                    10151972, 2234118, 3015516, 3380848, 13995, 735264], dtype=np.int32),

    }[dataset]

    if dataset == 'ScanNet' and version == '200':
        class_cnt = np.array([13291642, 2553010, 12907527, 1611300, 1615009, 1334219, 1186424, 1108679, 976358, 393724, 1356082, 219125,
                                120310, 227534, 1425879, 148487, 1062550, 210960, 891659, 409199, 364604, 259855, 202045, 210978, 93941,
                                547982, 79334, 179545, 126495, 134711, 109194, 223388, 81146, 16127, 191629, 1554295, 175261, 71670,
                                34269, 17081, 46971, 129209, 12381, 43423, 88600, 371356, 62312, 136545, 325679, 118661, 121474, 38283,
                                99468, 24096, 44750, 10155, 10394, 78171, 54611, 83397, 115206, 81559, 120124, 9143, 178780, 105484,
                                23721, 17070, 104484, 9437, 28365, 25483, 10770, 73828, 13436, 1318, 112754, 52712, 51209, 10108, 183481,
                                69395, 9548, 124397, 6580, 10372, 8589, 5324, 10370, 13446, 18881, 15248, 79518, 9475, 1039, 33281, 7730,
                                25418, 24794, 21637, 132270, 280833, 5304, 33869, 14338, 12724, 18543, 2876, 11949, 17356, 32527, 5689,
                                1293, 41149, 7186, 29846, 1660, 133360, 293239, 4511, 46786, 3376, 7944, 1946, 2396, 13967, 5709, 5013,
                                15186, 23950, 12317, 4111, 2396, 3238, 17667, 40539, 7930, 2133, 5023, 882, 3900, 374, 17873, 6387, 2810,
                                77133, 10664, 703, 1578, 3482, 505, 9307, 3438, 6170, 12782, 1194, 668, 1134, 2264, 3699, 2114, 9718,
                                1679, 133558, 1640, 8544, 708, 3681, 26471, 25258, 1768, 21646, 172214, 120005, 81209, 33143, 42457,
                                14354, 17011, 4014, 56782, 5594, 42526, 1462, 4073, 893, 21108, 540, 6572, 3951, 3854, 10796, 7699, 420,
                                455, 1429, 5887, 8057, 3748, 26968], dtype=np.int32)

    freq = class_cnt / float(sum(class_cnt))
    if weight_type == 'class':
        weight = 1 / (freq + 0.02)
    elif weight_type == 'classqrt':
        weight = 1 / np.sqrt(freq)
    elif weight_type == 'classlog':
        weight = - np.log(class_cnt) / np.log(class_cnt).sum()
    else:
        raise ValueError(f'not support weight type={weight_type}')

    if normalize == 'norm':
        weight = weight / weight.sum()
    elif normalize == 'unit':  # expectation E[w] = 1
        weight = weight / weight.sum() * len(weight)
    elif normalize:
        raise NotImplementedError(f'get_class_weight - not supported normalize = {normalize}')
    weight = tf.constant(weight, dtype=tf.float32)
    if labels is not None:
        weight = tf.gather(weight, labels)
    return weight

@tf_scope
def calc_loss(loss, labels, logits, config, smooth=None, num_classes=None, mask=None, stop_gradient=True, reduce_loss=True, raise_not_support=True):
    ncls = num_classes if num_classes else config.num_classes

    if stop_gradient:
        labels = tf.stop_gradient(labels)
    def masking(labels, logits, mask):
        labels = tf.boolean_mask(labels, mask) if mask is not None else labels
        logits = tf.boolean_mask(logits, mask) if mask is not None else logits
        return labels, logits

    if smooth:
        if labels.get_shape().as_list()[-1] != ncls:
            labels = tf.cast(tf.one_hot(labels, depth=ncls, axis=-1), tf.float32)
            labels = labels * (1 - float(smooth)) + (1 - labels) * float(smooth) / float(ncls - 1)
        else:
            labels_onehot = tf.cast(tf.one_hot(tf.argmax(labels, axis=-1), depth=ncls, axis=-1), tf.float32)
            labels = labels - labels_onehot * float(smooth) + (1 - labels_onehot) * float(smooth) / float(ncls - 1)

    if loss.startswith('softmax') or loss == 'xen':  # pixel-level loss
        # flatten to avoid explicit tf.assert in cross_entropy, which supports cpu only
        labels, logits = masking(labels, logits, mask)
        if len(labels.shape) == len(logits.shape):  # one-hot label
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]),
                                                                  logits=tf.reshape(logits, [-1, int(logits.shape[-1])]))
    elif loss.startswith('sigmoid') or loss == 'sig':  # pixel-level loss
        labels, logits = masking(labels, logits, mask)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.reshape(labels, [-1]), logits.dtype),
                                                       logits=tf.reshape(logits, [-1]))
    elif raise_not_support:
        raise NotImplementedError(f'not supported loss = {loss}')
    else:
        loss = None

    if loss is not None and reduce_loss:
        loss = tf.reduce_mean(loss)
    return loss

def calc_weight_mask(weight, inputs, stage_n, stage_i, ftype, config, mask=None, stop_gradient=None, cache=False):
    # larger the weight, larger the loss
    assert isinstance(weight, str)

    wtype = re.match('[a-zA-Z]+', weight)
    wtype = wtype.group(0) if wtype else ''
    wfloat = weight[len(wtype):]
    wfloat = float(wfloat) if wfloat else None
    stop_gradient = stop_gradient if stop_gradient is not None else wtype.endswith('Ng')

    key = f'{stage_n}{stage_i}-{wtype}'
    if cache and key in inputs['_glb']:
        return (inputs['_glb'][key], wfloat)

    wtype = wtype[:-2] if wtype.endswith('Ng') else wtype
    if wtype in ['gap', 'kl', 'xen']:
        # weighted by dist between pred (collected) - label (collected)
        probs = get_scene_features(inputs['head_dict']['result']['seg']['probs'], inputs, 'U0', (stage_n, stage_i), ftype, config, extend=False, name='probs')

        # [BxN] - weight calc based on labels-probs
        if wtype == 'gap':
            # gap on one-hot
            labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='max', extend=False)
            labels = tf.squeeze(labels, axis=-1) if labels is not None else inputs['point_labels']  # [BxN]
            if mask is not None:
                probs = tf.boolean_mask(probs, mask)
                labels = tf.boolean_mask(labels, mask)
            labels = tf.cast(tf.one_hot(labels, depth=config.num_classes, axis=-1), tf.float32)
            weight = 1 - tf.reduce_sum(labels * probs, axis=-1)
        else:
            # kl divergense as soft xen
            labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='soft', extend=False)
            labels = labels if labels is not None else tf.cast(tf.one_hot(inputs['point_labels'], depth=config.num_classes, axis=-1), tf.float32)  # [BxN, ncls]
            if mask is not None:
                probs = tf.boolean_mask(probs, mask)
                labels = tf.boolean_mask(labels, mask)
            xen = -tf.reduce_sum(tf.math.xlogy(labels, tf.maximum(probs, _eps)), axis=-1)
            if wtype == 'kl':
                xen += tf.reduce_sum(tf.math.xlogy(labels, labels), axis=-1)
            weight = xen

    elif wtype in ['lgt']:
        # soft xen between logits (per-branch) - labels (collected)
        f_dict = inputs['stage_list'][stage_n][stage_i]
        assert ftype == 'logits', f'not supported to interpreted as logits for ftype={ftype}'
        assert ftype in f_dict and f_dict[ftype] is not None, f'should have {ftype} built, but got at {stage_n}{stage_i}: {f_dict}'

        labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='soft', extend=False)
        labels = labels if labels is not None else tf.cast(tf.one_hot(inputs['point_labels'], depth=config.num_classes, axis=-1), tf.float32)  # [BxN, ncls]
        if mask is not None:
            logits = tf.boolean_mask(logits, mask)
            labels = tf.boolean_mask(labels, mask)
        weight = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name=wtype)

    elif wtype in ['cls']:
        # class-balancing weight
        ftype, _ = get_ftype(head_cfg.ftype.split('-')[-1])
        labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='max', extend=False)
        labels = tf.squeeze(labels, axis=-1) if labels is not None else inputs['point_labels']  # [BxN]
        if mask is not None:
            labels = tf.boolean_mask(labels, mask)
        weight = get_class_weight(config.dataset, labels, version=config.version, normalize={'s': '', 'N': 'norm', 'U': 'unit'}[dist[-1]])

    elif wtype == '':
        weight = None
    else:
        raise ValueError(f'not support wtype = {wtype} (original weight = {weight})')

    if stop_gradient and weight is not None:
        weight = tf.stop_gradient(weight)

    if cache:
        inputs['_glb'][key] = weight

    return weight, wfloat


# ---------------------------------------------------------------------------- #
# pred head & loss
# ---------------------------------------------------------------------------- #


class mlp_head(object):
    valid_weight = ['center', 'class', 'batch']  # valid spcial weight (str)

    def __call__(self, inputs, head_cfg, config, is_training):
        # assert head_cfg.ftype == None and head_cfg.stage == None
        drop = head_cfg.drop if head_cfg.drop else config.drop
        logits, labels, loss = mlp_head.pred(inputs, 'up', 0, head_cfg, config, is_training, drop=drop)
        latent = inputs['stage_list']['up'][0]['latent']
        probs = tf.nn.softmax(logits, axis=-1)
        inputs['stage_list']['up'][0]['probs'] = probs
        return {'loss': loss, 'latent': latent, 'logits': logits, 'probs': probs, 'labels': labels}

    @staticmethod
    @tf_scope
    def get_branch_head(rst_dict, head_cfg, config, is_training, fkey, drop=None, fdim=None, num_classes=None):
        # get features of current branch & storing into rst, upto probs
        ftype = get_ftype(fkey)[0]
        if ftype == 'f_out': return rst_dict['f_out']

        ftype_list = ['latent', 'logits', 'probs',]
        assert ftype in ftype_list, f'not supported ftype = {ftype} from fkey = {fkey}'
        if isinstance(drop, str):
            drop = re.search('\.\d+', drop)  # str float / None
            drop = float(drop.group()) if drop else None

        d_out = fdim if fdim is not None else head_cfg.d_out if head_cfg.d_out else config.first_features_dim
        if num_classes is None:
            num_classes = head_cfg.num_classes if head_cfg.num_classes else config.num_classes

        idx = max([ftype_list.index(k) + 1 if k in ftype_list else 0 for k in rst_dict])  # only consider fkey after existing keys
        if 'latent' not in rst_dict and ftype in ftype_list[idx:]:
            features = rst_dict['f_out']
            ops = head_cfg.mlp if head_cfg.mlp else fkey[len(ftype):] if fkey != ftype else 'mlp'
            kwargs = get_kwargs_mlp(head_cfg, config, is_training, _cnt=0)
            features = mlps_by_ops(features, d_out, d_out, ops=ops, kwargs=kwargs)
            rst_dict['latent'] = features
            idx += 1

        if drop is not None:
            features = rst_dict['latent']  # must have latent if applying dropout
            rst_dict['latent'] = dropout(features, rate=drop, is_training=is_training, name='dropout')

        if 'logits' not in rst_dict and ftype in ftype_list[idx:]:
            kwargs = get_kwargs(head_cfg, config, is_training)
            features = dense_layer(rst_dict['latent'], num_classes, f'linear', None, True, False, **kwargs)
            rst_dict['logits'] = features
            idx += 1

        if 'probs' not in rst_dict and ftype in ftype_list[idx:]:
            features = tf.nn.softmax(rst_dict['logits'], axis=-1)
            rst_dict['probs'] = features
            idx += 1

        return rst_dict[ftype]

    @staticmethod
    @tf_scope
    def pred(inputs, stage_n, stage_i, head_cfg, config, is_training, rst_dict=None, drop=None, labels='U0', labels_ext=False, reduce_loss=True):
        # TODO: may use rst_dict to pass on latent? - since may introduce mask-level supervision
        rst_dict = inputs['stage_list'][stage_n][stage_i] if rst_dict is None else rst_dict
        logits = mlp_head.get_branch_head(rst_dict, head_cfg, config, is_training, 'logits', drop=drop)

        # match pred (logits) to labels
        upsample_idx = get_sample_idx(inputs, (stage_n, stage_i), labels, head_cfg.ftype, config, kr=1)
        if upsample_idx is not None:
            logits = tf.squeeze(tf_gather(logits, tf.expand_dims(upsample_idx, axis=-1), get_mask=False), axis=-2)

        # temperature re-scaling
        if head_cfg.temperature:
            logits /= tf_get_variable('T', [], 'ones') if head_cfg.temperature == 'w' else float(head_cfg.temperature)

        # get labels
        label_n, label_i = parse_stage(labels, config.num_layers)[0] if isinstance(labels, str) else labels
        labels = get_scene_label(inputs, label_n, label_i, head_cfg.ftype, config, reduction='soft', extend=labels_ext)
        labels = labels if labels is not None else inputs['point_labels']  # [BxN, /num_classes]

        # correct by cloud labels mask
        if config.cloud_labels == 'multi':  # per-cloud labels
            num_multi = config.cloud_labels_multi
            mask_multi = np.zeros([len(num_multi), config.num_classes])
            for i, n in enumerate(num_multi):
                inds_start = num_multi[:i].sum()
                mask_multi[i][inds_start:inds_start + n] = 1
            mask_multi = tf.constant(mask_multi, dtype=tf.float32)  # [#cloud, ncls]

            cld_labels = tf.expand_dims(inputs['cloud_labels'], axis=-1)  # [B, 1]
            if 'batches_len' in inputs:
                batches_len = inputs['batches_len'][stage_i]
                Nmax = tf.reduce_max(batches_len)
                batch_mask = tf.sequence_mask(batches_len, Nmax)  # [B, Nmax] - bool mask
                cld_labels = tf.boolean_mask(cld_labels * tf.cast(batch_mask, tf.int32), batch_mask)  # [BxN] - per-point cloud label
            mask_multi = tf.gather(mask_multi, cld_labels)  # [BxN, ncls] - per-point mask on pred/logits
            logits -= _inf * (1 - mask_multi)
        elif config.cloud_labels:
            raise ValueError(f'not support cloud_labels={config.cloud_labels}')

        # save the full outputs
        full_labels = labels
        full_logits = logits

        # collect mask
        mask = None
        if len(config.ignored_labels) > 0:
            mask = tf.greater(tf.reduce_sum(labels, axis=-1), _eps) if len(labels.shape) == len(logits.shape) else tf.greater_equal(labels, 0)

        # match valid labels
        labels = tf.boolean_mask(labels, mask) if mask is not None else labels
        logits = tf.boolean_mask(logits, mask) if mask is not None else logits

        # calc loss
        loss = head_cfg.loss if head_cfg.loss else 'xen'  # default to softmax cross-entropy
        loss = calc_loss(loss, labels, logits, config, smooth=head_cfg.smooth, reduce_loss=False, raise_not_support=False)
        if loss is None and head_cfg.loss != 'none':
            raise NotImplementedError(f'not supported loss type = {head_cfg.loss}')

        # extra weight
        weight = None
        if isinstance(head_cfg.weight, float):
            weight = head_cfg.weight
        elif 'class' in head_cfg.weight:  # class weighting
            weight_type = [i for i in head_cfg.weight.split('-') if 'class' in i][0]
            weight = get_class_weight(config.dataset, labels, weight_type=weight_type, version=config.version)
            weight = tf.reshape(weight, [-1]) if loss._rank() == 1 else weight
        elif 'batch' in head_cfg.weight:  # batch-size weighting => mean inside cloud, then over batches
            weight = inputs['batch_weights']
            weight = tf.boolean_mask(weight, mask) if mask is not None else weight
        elif head_cfg.weight.startswith('w'):
            weight = float(head_cfg.weight[1:])
        elif head_cfg.weight:
            raise NotImplementedError(f'not supported weight = {head_cfg.weight}')

        loss = loss * weight if weight is not None else loss
        if reduce_loss:
            loss = tf.reduce_mean(loss)
        return full_logits, full_labels, loss


class cls_contrast_head(object):

    @staticmethod
    @tf_scope
    def collect_target_cur(features, labels, n, i, inputs, head_cfg, config, is_training):
        ncls = config.num_classes
        ftype, ptype = get_ftype(head_cfg.ftype.replace('seg', ''))
        d_out = int(features.shape[-1])
        labels_onehot = labels if len(labels.shape) == len(features.shape) else tf.one_hot(labels, depth=ncls, axis=-1)  # invalid label (-1) becomes all-zeros

        # prepare features - target
        with tf.variable_scope('mask'):
            target_mask = None
            if config.ignored_labels:
                target_mask_ign = tf.reshape(labels, [-1]) >= 0
                target_mask = tf.logical_and(target_mask, target_mask_ign) if target_mask is not None else target_mask_ign

        target_feat = tf.reshape(features, [-1, d_out])  # [BN, d_out]
        target_label = tf.reshape(labels_onehot, [-1, ncls])  # [BN, ncls] - per-cls mask
        # - apply mask-weight
        if target_mask is not None:
            target_feat = tf.boolean_mask(target_feat, target_mask)
            target_label = tf.boolean_mask(target_label, target_mask)
        target_cnt = tf.reduce_sum(target_label, axis=0)  # [ncls] - #point of each cls
        target_cls_mask = target_cnt > 0
        target_reduce = head_cfg.target_reduce

        # - apply reduce
        with tf.variable_scope('reduce'):
            target_label = tf.cast(target_label, tf.float32)  # [BN, ncls]
            target_feat = tf.expand_dims(target_label, axis=-1) * tf.expand_dims(target_feat, axis=-2)  # [BN, ncls, d]
            target_feat = tf.reduce_sum(target_feat, axis=0)  # [ncls, d]
            if target_reduce == 'mean':  # default
                target_feat = target_feat / (tf.expand_dims(target_cnt, axis=-1) + _eps)
        return target_feat, target_label, target_mask, target_cls_mask, target_cnt

    @staticmethod
    @tf_scope
    def momentum_target(features, target_feat, target_cls_mask, inputs, n, i, head_cfg, config, is_training):
        ni = f'{n}{i}'
        ncls = config.num_classes
        d_out = int(features.shape[-1])

        mom_dict = {}
        target_feat_cur = target_feat
        target_cls_mask_cur = target_cls_mask
        # target features - by mom update
        # target_feat = tf.stop_gradient(target_feat)
        trainable = False
        init = get_initializer(head_cfg.momentum_init if head_cfg.momentum_init else config.init)
        target_feat_mom = tf.get_variable(f'{ni}_fcls', shape=[ncls, d_out], initializer=init, trainable=trainable)  # shared across devices - default
        target_cls_mask = tf.cast(target_cls_mask, tf.float32)

        def _update():
            # conditional update - training phase only
            mom_avg = float(head_cfg.momentum_update)
            mom_avg = tf.expand_dims(mom_avg * target_cls_mask, axis=-1)  # [ncls, 1] - update features for only presented cls
            if head_cfg.momentum_update_stage_mask:
                mom_avg += tf.expand_dims(1 - target_cls_mask, axis=-1)
            target_feat_update = mom_avg * target_feat_mom + (1 - mom_avg) * target_feat_cur  # updated mom features
            with tf.control_dependencies([tf.assign(target_feat_mom, target_feat_update)]):
                target_feat_update = tf.identity(target_feat_update)
            return target_feat_update

        if head_cfg.momentum_update_stage.startswith('glb_'):
            # enable cross-device update
            # NOTE - solve for update later
            target_feat_update = target_feat_mom  # update entry - preserved in mom_dict
            mom_dict['target_cls_mask'] = target_cls_mask
        else:
            cond = is_training
            # if target_mask is not None:
            #     cond = tf.logical_and(cond, tf.reduce_any(target_mask))
            target_feat_update = tf_cond(cond, true_fn=_update, false_fn=lambda: target_feat_mom)
            target_feat_update = tf.stop_gradient(target_feat_update)
        mom_dict[f'{ni}-f_out'] = target_feat_update

        target_feat = target_feat_update
        target_cls_mask = None
        return target_feat, target_cls_mask, mom_dict

    @staticmethod
    @tf_scope
    def collect_target(features, labels, n, i, inputs, head_cfg, config, is_training, return_dict=False):
        ni = f'{n}{i}'
        ncls = config.num_classes
        ftype, ptype = get_ftype(head_cfg.ftype.replace('seg', ''))
        d_out = int(features.shape[-1])
        # labels_onehot = labels if len(labels.shape) == len(features.shape) else tf.one_hot(labels, depth=ncls, axis=-1)  # invalid label (-1) becomes all-zeros

        # prepare features - target
        collect_func = cls_contrast_head.collect_target_cur
        target_feat, target_label, target_mask, target_cls_mask, target_cnt = collect_func(features, labels, n, i, inputs, head_cfg, config, is_training)

        mom_dict = {}
        target_feat_cur = target_feat  # [ncls, d]
        target_cls_mask_cur = target_cls_mask  # [ncls]
        if head_cfg.momentum_update:
            # prepare features - target - momentum update
            target_feat, target_cls_mask, mom_dict = cls_contrast_head.momentum_target(features, target_feat, target_cls_mask, inputs, n, i, head_cfg, config, is_training, name='momentum')

        if return_dict:
            return {
                'feat': target_feat,
                'feat_cur': target_feat_cur,
                'label': target_label,
                'mask': target_mask,
                'cnt': target_cnt,
                'cls_mask': target_cls_mask,
                'cls_mask_cur': target_cls_mask_cur,
                'mom_dict': mom_dict
            }
        return target_feat, target_feat_cur, target_cls_mask, target_cls_mask_cur, mom_dict


class pseudo_head(object):

    @staticmethod
    def get_stage_info(n, i, inputs, head_cfg, config, is_training, get_upsample=True, features=None):
        ni = f'{n}{i}'
        ftype, ptype = get_ftype(head_cfg.ftype.replace('seg', ''))

        # fetch features - project shared for all cls contrast
        if features is not None:
            pass
        elif head_cfg.ftype.startswith('seg'):
            assert n == 'up' and i == 0
            ni = f'seg-{n}{i}'
            features = inputs['head_dict']['result']['seg'][ftype]
        else:
            features = mlp_head.get_branch_head(inputs['stage_list'][n][i], head_cfg, config, is_training, ftype, name=f'{ni}-f')  # [BxN, d]
        origin_features = features

        d_out = int(features.shape[-1])
        if head_cfg.project_fdim:
            d_out = int(head_cfg.project_fdim)

        if head_cfg.project:
            proj_ops = head_cfg.project.split('-') if isinstance(head_cfg.project, str) else head_cfg.project
            proj_ops = [i for i in proj_ops if i]
            proj_kwargs = get_kwargs_mlp(head_cfg, config, is_training)

            for ops in proj_ops:
                if 'mlp' in ops or 'lin' in ops:
                    d_mid = d_out
                    if re.search('r\d+$', ops):
                        r = re.search('r\d+$', ops).group()
                        ops = ops[:-len(r)]
                        d_mid = int(d_out / int(r[1:]))
                    features = mlps_by_ops(features, d_mid, d_out, ops, kwargs=proj_kwargs, name=f'{ni}-proj')
                elif any(ops.startswith(i) for i in ['avg', 'max']):
                    ops, kr = ops[:3], ops[3:]
                    with tf.variable_scope(f'{ni}-proj-{ops}'):
                        _, _, neighbor_idx = fetch_supports(inputs, n, i, 'out')
                        if kr and int(kr) > int(neighbor_idx.shape[-1]):
                            from ops import get_tf_func
                            search_func = get_tf_func(config.search)
                            pts = inputs['points'][i]
                            args = [inputs['batches_len'][i], inputs['batches_len'][i], int(kr)] if 'batches_len' in inputs else [int(kr)]
                            neighbor_idx = search_func(pts, pts, *args, device='/cpu:0')
                        neighbor_idx = neighbor_idx[..., :int(kr)] if kr else neighbor_idx
                        neighbor_f, neighbor_m = tf_gather(features, neighbor_idx)  # [BxN, k, d]
                        features = apply_reduction(neighbor_f, mask=neighbor_m, reduction=ops)
                else:
                    raise ValueError(f'not support project ops = {ops} from {head_cfg.project}')
        assert features.shape[-1] == d_out, f'specify fdim={head_cfg.project_fdim} but got {features} (project={head_cfg.project})'

        # match features
        upsample_idx = None
        if i != 0 and get_upsample:
            upsample_idx = get_sample_idx(inputs, (n, i), 'U0', ftype, config, kr=1)  # nearest - [BxN]
            upsample_idx = tf.expand_dims(upsample_idx, axis=-1)

        # fetch labels
        labels = inputs['point_labels']
        return features, labels, origin_features, upsample_idx

    @staticmethod
    @tf_scope
    def _calc_loss(loss_n, pseudo_label, pseudo_logits, logits, head_cfg, config):
        ncls = config.num_classes
        _loss_n = loss_n
        if loss_n.startswith('js'):
            probs = tf.nn.softmax(logits, axis=-1)
            m = (pseudo_label + probs) / 2
            loss = tf.reduce_sum(tf.math.xlogy(pseudo_label, pseudo_label / tf.maximum(m, _eps)) + tf.math.xlogy(probs, probs / tf.maximum(m, _eps)), axis=-1)
        elif loss_n == 'kl':
            probs = tf.nn.softmax(logits, axis=-1)
            loss = calc_dist(pseudo_label, probs, dist=loss_n, keepdims=False)
        elif loss_n.startswith('klr'):  # kl(q||p)
            probs = tf.nn.softmax(logits, axis=-1)
            loss = calc_dist(probs, pseudo_label, dist='kl', keepdims=False)
        elif loss_n.startswith('mse'):  # mse
            probs = tf.nn.softmax(logits, axis=-1)
            loss = calc_dist(probs, pseudo_label, dist='l2square', keepdims=False) / 2
        else:
            loss = calc_loss(loss_n, pseudo_label, logits, config, name='loss', stop_gradient=False, reduce_loss=False)  # [BN]
        if config.debug:
            print(tf.get_variable_scope().name, _loss_n, loss, 'pseudo/logits', pseudo_label, logits)
        return loss

    @staticmethod
    @tf_scope
    def calc_loss(pseudo_label, pseudo_logits, logits, mask, inputs, head_cfg, config):
        loss = pseudo_head._calc_loss(head_cfg.loss, pseudo_label, pseudo_logits, logits, head_cfg, config)
        reduce_fn = tf.reduce_mean
        loss = reduce_fn(loss)
        if mask is not None:
            loss = tf_cond(tf.reduce_any(mask), true_fn=lambda: loss, false_fn=lambda: 0.0)
        return loss

    @staticmethod
    @tf_scope
    def prop_label_cls(inputs, head_cfg, config, is_training, n='up', i=0):
        # n, i = 'up', 0
        ni = f'{n}{i}'
        ncls = config.num_classes
        ftype, ptype = get_ftype(head_cfg.ftype)

        features, labels, origin_f, upsample_idx = pseudo_head.get_stage_info(n, i, inputs, head_cfg, config, is_training)
        if upsample_idx is not None:
            features = tf.squeeze(tf_gather(features, upsample_idx, shadow_fn=None, get_mask=False), axis=-2)
            origin_f = tf.squeeze(tf_gather(origin_f, upsample_idx, shadow_fn=None, get_mask=False), axis=-2)
        d_out = int(features.shape[-1])

        # labeled pts
        # NOTE: need to re-write collect target if wanting to control on momentum - e.g. including features of unlabeled pts in update via confidence/dist
        features_target = origin_f if head_cfg.project_stage == 'post' else features
        target_dict = cls_contrast_head.collect_target(features_target, labels, n, i, inputs, head_cfg, config, is_training, name='target', return_dict=True)
        target_feat, target_feat_cur, target_cls_mask, mom_dict = [target_dict[k] for k in ['feat', 'feat_cur', 'cls_mask', 'mom_dict']]
        target_feat_mom = mom_dict[f'{ni}-f_out'] if f'{ni}-f_out' in mom_dict else None

        # unlabeled pts
        sample_mask = None
        # TODO: may further distinguish invalid vs nolabel pts
        sample_mask = labels < 0 if config.ignored_labels and config.weak_supervise else None  # [BxN]
        sample_feat = features

        # calc dist
        if 'norm' in head_cfg.dist or head_cfg.dist == 'cos':
            target_feat = tf.nn.l2_normalize(target_feat, axis=-1, epsilon=_eps)  # [ncls, d] / [BxN, ncls, d]
            sample_feat = tf.nn.l2_normalize(sample_feat, axis=-1, epsilon=_eps)
        # [BxN, ncls] - (small <- similar, dissim -> inf)
        if len(target_feat.shape) == 2 and target_feat.get_shape().as_list()[0] is None and 'dot' in head_cfg.dist:
            dist = -tf.matmul(sample_feat, tf.transpose(target_feat, perm=[1, 0]))
        else:
            dist = calc_dist(tf.expand_dims(sample_feat, axis=-2), tf.expand_dims(target_feat, axis=-3) if len(target_feat.shape) == 2 else target_feat, dist=head_cfg.dist, keepdims=False, name='dist')

        # revert & scale the dist - (small <- dissim, similar -> large)
        dist_scaled = dist
        scale = head_cfg.scale
        if scale == 'inv':
            dist_scaled = 1 / (dist_scaled + _eps)  # may have inf - if using dot/normdot
        elif scale == 'exp':
            dist_scaled = tf.exp(-dist_scaled)
        elif scale == 'negexp':
            dist_scaled = -tf.exp(dist_scaled)
        elif scale == 'log':
            dist_scaled = -tf.log(dist_scaled + _eps)  # need dist > 0
        elif scale == 'neg':
            dist_scaled = -dist_scaled
        else:
            raise ValueError(f'not support scale=\'{scale}\'')
        dist_scaled = tf.clip_by_value(dist_scaled, -_inf, _inf)  # [BxN, ncls]

        # normalize into soft label
        if target_cls_mask is not None:
            target_cls_mask = tf.expand_dims(target_cls_mask, axis=0)  # [1, ncls]
            if 'batches_len' not in inputs: target_cls_mask = tf.expand_dims(target_cls_mask, axis=0)  # [1, 1, ncls]
        dist_norm = normalize(dist_scaled, head_cfg.norm, axis=-1, mask=target_cls_mask)  # [BxN, ncls]

        logits = inputs['head_dict']['result']['seg']['logits']
        pseudo_label = dist_norm
        pseudo_logits = dist_scaled

        # calc loss
        if not head_cfg.coadapt:
            pseudo_label = tf.stop_gradient(pseudo_label)
            pseudo_logits = tf.stop_gradient(pseudo_logits)
        # logits = inputs['head_dict']['result']['seg']['logits'] if 'seg' in inputs['head_dict']['result'] else \
        #     mlp_head.get_branch_head(inputs['stage_list']['up'][0], head_cfg, config, is_training, 'logits', drop)

        # masking
        if sample_mask is not None:
            logits = tf.boolean_mask(logits, sample_mask)
            pseudo_label = tf.boolean_mask(pseudo_label, sample_mask)  # [BN, d]
            pseudo_logits = tf.boolean_mask(pseudo_logits, sample_mask)

        mom_f = None
        if head_cfg.momentum_update_stage:
            # enable cross-device update 
            _feat_mom = mom_dict[f'{ni}-f_out']
            _cls_mask = mom_dict.pop('target_cls_mask')
            mom_inputs = inputs['momentum_dict'] if 'momentum_dict' in inputs else {}
            if head_cfg.head_n not in mom_inputs:
                mom_inputs[head_cfg.head_n] = {}
            mom_inputs[head_cfg.head_n].update(mom_dict)  # update entry - preserved in current mom_dict
            inputs['momentum_dict'] = mom_inputs
            target_feat_cur = tf_cond(is_training, true_fn=lambda: target_feat_cur, false_fn=lambda: _feat_mom)
            mom_dict = {f'{ni}-f_out': target_feat_cur}  # device output - current

        loss = pseudo_head.calc_loss(pseudo_label, pseudo_logits, logits, sample_mask, inputs, head_cfg, config)
        if head_cfg.weight:
            loss *= float(head_cfg.weight)

        head_dict = {
            'loss': loss,
            'f_out': features,
            # 'latent': dist,
            'logits': dist_scaled,
            'probs': dist_norm,  # full distance as probs - availble in val
            'labels': pseudo_label,  # may not available in val, due to sample_mask
        }
        if mom_dict:
            head_dict['momentum_dict'] = mom_dict
        if config._weak_supervise_reserve and sample_mask is not None:
            head_dict['sample_mask'] = sample_mask  # mask for pseudo labels dst
            head_dict['target_feat'] = target_feat
        return head_dict

    def __call__(self, inputs, head_cfg, config, is_training):
        func = pseudo_head.prop_label_cls
        return func(inputs, head_cfg, config, is_training)


def get_head_ops(head_n, raise_not_found=True):

    # head_n == idx_name_pre
    if head_n == 'mlp':
        head_ops = mlp_head()

    elif head_n == 'pseudo':
        head_ops = pseudo_head()

    # raise or skip
    elif raise_not_found:
        raise NotImplementedError(f'not supported head_n = {head_n}')
    else:
        head_ops = None
    return head_ops

def apply_head_ops(inputs, head_cfg, config, is_training):
    head_ops = get_head_ops(head_cfg.head_n)
    rst = head_ops(inputs, head_cfg, config, is_training)
    return rst
