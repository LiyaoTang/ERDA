import sys
import numpy as np
from sklearn.metrics import confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.sum / self.count

class Metrics(dict):
    def __init__(self, *args, scale=1, order=['mIoU', 'mIoU_cld', 'OA', 'mACC'], task=None, **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        self.scale = scale
        self.order = [order] if isinstance(order, str) else list(order)  # the importance rank of metrics - main key = order[0]
        self._scalar_to_list = {'mIoU': 'IoUs', 'mACC': 'ACCs'}

        main_list = [self._scalar_to_list[i] for i in self.order if i in self._scalar_to_list]
        self.main_list = main_list[0] if main_list else None

        if task in ['seg', 'segmentation']:
            self.seg()
        elif task in ['cls', 'classification']:
            self.cls()
        elif task:
            raise ValueError(f'Metrics not support predefined task={task}')
        return

    # def __missing__(self, key):
    #     return None

    def cls(self):
        self.order = ['mACC', 'OA']
        self.main_list = 'ACCs'
        return self

    def seg(self):
        self.order = ['mIoU', 'mIoU_cld', 'OA', 'mACC']
        self.main_list = 'IoUs'
        return self

    # Comparison
    # ------------------------------------------------------------------------------------------------------------------

    def _is_valid(self, other, raise_invalid=True):
        if self.order[0] not in other:
            if raise_invalid:
                raise ValueError(f'missing main key - {self.order[0]}, in order {self.order}')
            return False
        return True

    def __eq__(self, other):  # care only the main key
        self._is_valid(self)
        self._is_valid(other)
        return self[self.order[0]] == other[self.order[0]]

    def __gt__(self, other):
        self._is_valid(self)
        self._is_valid(other)
        for k in self.order:
            if k not in self:  # skip if not available
                continue
            if k not in other or self[k] > other[k]:  # True if more completed
                return True
            elif self[k] < other[k]:
                return False

        # all equal (at least for main key)
        return False

    # Pretty print
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def scalar_str(self):
        scalar_m = [k for k in self.order if k in self and self[k]]
        s = ''.join([f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m])
        return s
    @property
    def list_str(self):
        if self.main_list is None:
            return ''
        list_m = [k for k in [self.main_list] if k in self and self[k] is not None]
        s = []
        for k in list_m:
            m = self.list_to_line(k)
            s += [m]
        s = ' | '.join(s)
        return s
    @property
    def final_str(self):
        s = str(self)
        s = ['-' * len(s), s, '-' * len(s)]
        if 'ACCs' in self:
            s = ['ACCs = ' + self.list_to_line('ACCs')] + s
        return '\n'.join(s)
 
    def print(self, full=True, conf=True):
        s = self.full() if full else self.final_str
        if conf and 'conf' in self:
            conf = self['conf']
            # assert np.issubdtype(conf.dtype, np.integer)
            with np.printoptions(linewidth=sys.maxsize, threshold=sys.maxsize, precision=3):
                print(self['conf'])
        print(s)

    def full(self, get_list=False, keys=None):
        # separate line print each group of metrics
        scalar_m = [k for k in ['OA', 'mACC', 'mIoU_cld', 'mIoU'] if k in self and self[k] and k in self.order]

        str_d = {k: f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m}  # scalar_m -> str
        for k_scalar, k_list in self._scalar_to_list.items():
            if k_scalar not in str_d: continue
            str_d[k_scalar] += ' | ' + self.list_to_line(k_list)

        max_len = max(len(v) for v in str_d.values())
        s = ['-' * max_len, *[v for v in str_d.values()], '-' * max_len]
        s = s if get_list else '\n'.join(s)
        return s

    def __repr__(self):
        return ' | '.join([k for k in [self.scalar_str, self.list_str] if k])

    def list_to_line(self, k):
        l = k if isinstance(k, list) else self[k] if k in self else None
        m = ' '.join([f'{i/self.scale*100:<5.2f}' for i in l]) if l is not None else ''
        return m


def classification_metrics(preds, targets, num_classes):
    seen_class = [0.0 for _ in range(num_classes)]
    correct_class = [0.0 for _ in range(num_classes)]
    preds = np.argmax(preds, -1)
    correct = np.sum(preds == targets)
    seen = preds.shape[0]
    for l in range(num_classes):
        seen_class[l] += np.sum(targets == l)
        correct_class[l] += (np.sum((preds == l) & (targets == l)))
    acc = 1.0 * correct / seen
    avg_class_acc = np.mean(np.array(correct_class) / np.array(seen_class))
    return acc, avg_class_acc


def metrics_from_confusions(confusions, proportions=None):
    """
    Computes IoU from confusion matrices.
    Args:
        confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes; gt (row) x pred (col).
    """

    confusions = confusions.astype(np.float32)
    if proportions is not None:
        # Balance with real proportions
        confusions *= np.expand_dims(proportions.astype(np.float32) / (confusions.sum(axis=-1) + 1e-6), axis=-1)

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)
    ACC = TP / (TP_plus_FN + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)
    mACC = np.sum(ACC, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later, or simply denotes absence with nan
    IoU += mask * mIoU
    # IoU[mask] = float('nan')
    ACC[mask] = float('nan')

    # Compute Accuracy
    OA = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)
    m = {
        'mIoU': mIoU.mean(),
        'mACC': mACC.mean(),
        'OA': OA,
        'IoUs': IoU,
        'ACCs': ACC,
        '_valid_mask': np.logical_not(mask),  # valid mask
    }
    m = Metrics(m)
    return m


def metrics_from_result(preds, labels, num_classes, label_to_idx=None, proportions=None, projections=None, keys=None):
    """
    list of pred-label
    """
    confs = []
    num_classes = np.arange(num_classes) if isinstance(num_classes, int) else list(num_classes)
    projections = projections if projections is not None else [None] * len(preds)
    for cur_pred, cur_label, cur_proj in zip(preds, labels, projections):
        if cur_proj is not None:  # re-project
            cur_pred = cur_pred[cur_proj]
        if len(cur_pred.shape) > 1:  # prob matrix
            cur_pred = np.argmax(cur_pred, axis=-1).astype(int)
        if label_to_idx is not None:  # match to the preds
            cur_label = label_to_idx[cur_label].astype(int)
            if np.any(cur_label < 0):  # potential invalid label position (would be ignored by specifying labels anyway)
                valid_mask = cur_label >= 0
                cur_pred = cur_pred[valid_mask]
                cur_label = cur_label[valid_mask]
        cur_conf = confusion_matrix(cur_label, cur_pred, labels=num_classes)
        confs.append(cur_conf)

    confs = np.array(confs)
    conf = confs.sum(axis=0)

    m = metrics_from_confusions(conf, proportions=proportions)
    m['conf'] = conf
    m['confs'] = confs
    return m


def multi_metrics_from_result(preds, labels, cloud_labels, num_classes, cloud_labels_multi, label_values, multi_type, multi_main=None, label_to_idx=None, proportions=None, projections=None, remove_cls0=False):
    multi_type = multi_type.lower()
    assert multi_type in ['shapenet', 'partnet']

    confs = []
    cloud_labels_multi = np.array(cloud_labels_multi).copy()
    num_classes = np.arange(num_classes) if isinstance(num_classes, int) else list(num_classes)
    projections = projections if projections is not None else [None] * len(preds)

    cls0_inds = np.cumsum([0] + list(cloud_labels_multi))[:-1] if remove_cls0 else None
    for cur_pred, cur_label, cur_proj in zip(preds, labels, projections):
        if cur_proj is not None:  # re-project
            cur_pred = cur_pred[cur_proj]
        if remove_cls0:  # remove cls-0
            assert len(cur_pred.shape) > 1
            cls0_mask = (cur_labels[:, None] != cls0_inds[None, :]).all(axis=-1)  # 0-label points
            cur_labels = cur_labels[cls0_mask]
            cur_pred = cur_pred[cls0_mask]
            cur_pred[:, cls0_inds] = -np.inf
        if len(cur_pred.shape) > 1:  # prob matrix
            cur_pred = np.argmax(cur_pred, axis=-1).astype(int)
        if label_to_idx is not None:  # match to the preds
            cur_label = label_to_idx[cur_label].astype(int)
            if np.any(cur_label < 0):  # potential invalid label position (would be ignored by specifying labels anyway)
                valid_mask = cur_label >= 0
                cur_pred = cur_pred[valid_mask]
                cur_label = cur_label[valid_mask]
        cur_conf = confusion_matrix(cur_label, cur_pred, labels=num_classes)
        confs.append(cur_conf)
    confs = np.array(confs)

    if proportions is not None:  # balance with real proportions
        confusions *= np.expand_dims(proportions.astype(np.float32) / (confusions.sum(axis=-1) + 1e-6), axis=-1)
    if remove_cls0:
        cloud_labels_multi -= 1

    if multi_type == 'shapenet':
        # get per-cloud-type & per-cloud mIoU
        cld_m = []  # per-cloud metrics
        for conf, cld_l in zip(confs, cloud_labels):
            n_start = cloud_labels_multi[:cld_l].sum()
            n_end = n_start + cloud_labels_multi[cld_l]
            conf = conf[n_start:n_end, n_start:n_end]
            m = metrics_from_confusions(conf)
            cld_m.append(m)
        mIoUs = np.array([m['mIoU'] for m in cld_m])  # per-cloud mIoU
        IoUs_multi = np.array([mIoUs[l == cloud_labels].mean() for l in label_values])  # per-cloud-type mIoU
        mIoU_multi = IoUs_multi.mean()  # mIoU over cloud-type

    elif multi_type == 'partnet':
        # per-cloud-type mIoU on joined clouds
        m = metrics_from_confusions(confs.sum(axis=0))
        IoUs_multi = []
        for i, ncls_cld in enumerate(cloud_labels_multi):
            n_start = cloud_labels_multi[:i].sum()
            n_end = n_start + ncls_cld
            IoUs_multi.append((m['IoUs'] * m['_valid_mask'])[n_start:n_end].mean())
        IoUs_multi = np.array(IoUs_multi)
        mIoU_multi = IoUs_multi.mean()

        # per-cloud-type mIoU from separated clouds
        mIoUs = []  # per-cloud mIoU
        mIoUs_mask = []
        for conf, cld_l in zip(confs, cloud_labels):
            n_start = cloud_labels_multi[:cld_l].sum()
            n_end = n_start + cloud_labels_multi[cld_l]
            conf = conf[n_start:n_end, n_start:n_end]
            tp, tp_fp, tp_fn = conf.diagonal(), conf.sum(axis=-2), conf.sum(axis=-1)
            IoUs = tp / (tp_fp + tp_fn - tp)
            mIoUs.append(IoUs[np.isfinite(IoUs)].mean())
            mIoUs_mask.append(np.isfinite(IoUs).any())
        mIoUs, mIoUs_mask = np.array(mIoUs), np.array(mIoUs_mask)
        IoUs = np.array([mIoUs[l == cloud_labels][np.array(mIoUs_mask)[l == cloud_labels]].mean() for l in label_values])
        mIoUs = IoUs

    else:
        raise ValueError(f'not support multi_type={multi_type}')

    m = Metrics({'mIoU': mIoU_multi, 'IoUs': IoUs_multi, 'mIoU_cld': mIoUs.mean()})
    if multi_main in ['multi_cld', 'cld']:
        m.order = ['mIoU_cld', 'mIoU', 'OA', 'mACC']
    else:
        assert not multi_main, f'not support multi_main={multi_main}'
    return m


def metrics(confusions, ignore_unclassified=False):
    """
    Computes different metrics from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
    """

    # If the first class (often "unclassified") should be ignored, erase it from the confusion.
    if (ignore_unclassified):
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FP = np.sum(confusions, axis=-1)
    TP_plus_FN = np.sum(confusions, axis=-2)

    # Compute precision and recall. This assume that the second to last axis counts the truths (like the first axis of
    # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    PRE = TP / (TP_plus_FN + 1e-6)
    REC = TP / (TP_plus_FP + 1e-6)

    # Compute Accuracy
    ACC = np.sum(TP, axis=-1) / (np.sum(confusions, axis=(-2, -1)) + 1e-6)

    # Compute F1 score
    F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

    # Compute IoU
    IoU = F1 / (2 - F1)

    return PRE, REC, F1, IoU, ACC


def smooth_metrics(confusions, smooth_n=0, ignore_unclassified=False):
    """
    Computes different metrics from confusion matrices. Smoothed over a number of epochs.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param smooth_n: (int). smooth extent
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) precision, recall, F1 score, IoU score
    """

    # If the first class (often "unclassified") should be ignored, erase it from the confusion.
    if ignore_unclassified:
        confusions[..., 0, :] = 0
        confusions[..., :, 0] = 0

    # Sum successive confusions for smoothing
    smoothed_confusions = confusions.copy()
    if confusions.ndim > 2 and smooth_n > 0:
        for epoch in range(confusions.shape[-3]):
            i0 = max(epoch - smooth_n, 0)
            i1 = min(epoch + smooth_n + 1, confusions.shape[-3])
            smoothed_confusions[..., epoch, :, :] = np.sum(confusions[..., i0:i1, :, :], axis=-3)

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(smoothed_confusions, axis1=-2, axis2=-1)
    TP_plus_FP = np.sum(smoothed_confusions, axis=-2)
    TP_plus_FN = np.sum(smoothed_confusions, axis=-1)

    # Compute precision and recall. This assume that the second to last axis counts the truths (like the first axis of
    # a confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    PRE = TP / (TP_plus_FN + 1e-6)
    REC = TP / (TP_plus_FP + 1e-6)

    # Compute Accuracy
    ACC = np.sum(TP, axis=-1) / (np.sum(smoothed_confusions, axis=(-2, -1)) + 1e-6)

    # Compute F1 score
    F1 = 2 * TP / (TP_plus_FP + TP_plus_FN + 1e-6)

    # Compute IoU
    IoU = F1 / (2 - F1)

    return PRE, REC, F1, IoU, ACC

def partnet_metrics(num_classes, num_parts, objects, preds, targets, verbose=False):
    shape_iou_tot = [0.0] * num_classes  # #cloud types
    shape_iou_cnt = [0] * num_classes

    part_intersect = [np.zeros((num_parts[o_l]), dtype=np.float32) for o_l in range(num_classes)]  # per-cloud-type #cls
    part_union = [np.zeros((num_parts[o_l]), dtype=np.float32) + 1e-6 for o_l in range(num_classes)]

    shape_mious = []
    for obj, cur_pred, cur_gt in zip(objects, preds, targets):
        cur_gt = np.squeeze(cur_gt)
        cur_num_parts = num_parts[obj]  # per-cloud cls - cloud_labels_multi
        cur_pred = np.argmax(cur_pred[:, 1:], axis=-1) + 1
        cur_pred[cur_gt == 0] = 0
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        cur_shape_ious = np.zeros(cur_num_parts)
        cur_shape_conf = confusion_matrix(cur_gt, cur_pred, labels=np.arange(cur_num_parts))
        for j in range(1, cur_num_parts):
            cur_gt_mask = (cur_gt == j)
            cur_pred_mask = (cur_pred == j)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)
            cur_shape_ious[j] = np.sum(cur_gt_mask & cur_pred_mask) / np.sum(cur_gt_mask | cur_pred_mask)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[obj][j] += intersect
                part_union[obj][j] += union
        if cur_shape_iou_cnt > 0:
            cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt  # current cloud mIoU
            shape_iou_tot[obj] += cur_shape_miou  # store into its cloud type
            shape_iou_cnt[obj] += 1
        shape_mious.append([cur_shape_miou, cur_shape_iou_tot, cur_shape_iou_cnt, cur_shape_ious, cur_shape_conf])

    msIoU = np.array([shape_iou_tot[o_l] / shape_iou_cnt[o_l] for o_l in range(num_classes)])  # per-cloud-type mIoU (over each separated cloud)
    if verbose:
        print('msIoU\n' + ' '.join(f'{i*100:.1f}' for i in msIoU))
    if verbose > 1:
        print('\t cnt - ', shape_iou_cnt)
        print('all shape mious\n' + ' '.join([f'{i[0]*100:.1f}' for i in shape_mious]))
        print('all shape cnt\n' + ' '.join([f'{i[2]}' for i in shape_mious]))

    part_iou = np.array([np.divide(part_intersect[o_l][1:], part_union[o_l][1:]) for o_l in range(num_classes)])  # per- [cloud-type x per-cloud-type-cls] IoU (remove 0-label) - summed over same-type clouds
    mpIoU = np.array([np.mean(part_iou[o_l]) for o_l in range(num_classes)])  # per-cloud-type mIoU (over joined clouds)
    if verbose:
        print('part_iou\n' + ' '.join(f'{i*100:.1f}' for i in np.concatenate(part_iou, axis=0)))
        mmsIoU = np.mean(np.array(msIoU)) * 100
        mmpIoU = np.mean(mpIoU) * 100
        print(f'mmsIoU={mmsIoU}, mmpIoU={mmpIoU}')
        # i = 0
        # print(f'[i={i}] ious ', ' '.join([f'{i*100:.1f}' for i in shape_mious[i][3]]), '=> confs', shape_mious[i][4].shape)
        # with np.printoptions(linewidth=sys.maxsize, threshold=sys.maxsize, precision=3):
        #     print(shape_mious[i][4])

    return msIoU, mpIoU
