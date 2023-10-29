import numpy as np
import tensorflow as tf
from functools import partial

def farthest_sample(points, n):
    """
    points : [B, N, 3]  -   support points
    n      : int        -   num of points to be sampled
    """
    shape = tf.shape(points)
    B = shape[0]
    N = shape[1]

    select = tf.zeros(tf.stack([B, 1]), dtype=tf.int64)  # [B, 1]  - random select the first point as start
    # select_f = tf.zeros(tf.stack([B, 1]), dtype=tf.bool)
    # select_m = tf.ones(tf.stack([B, N - 1]), dtype=tf.bool)  # [B, N]  - valid mask (T/F - can/not select)
    # select_m = tf.concat([select_f, select_m], axis=-1)

    dist_N = tf.ones(tf.stack([B, N]), dtype=tf.float32) * float('inf')  # [B, N] - init dist
    # dist_i = tf.cast(tf.range(N), dtype=tf.int64)
    # dist_b = tf.cast(tf.range(B), dtype=tf.int64)

    def body_gather(idx, dist_N, select, select_m):  # idx = 0 -> n
        update_i = select[..., -1:]  # [B, 1] - the selected idx
        # update_i = tf.gather(select, [idx], axis=1)
        update_p = tf.gather(points, update_i, batch_dims=1)  # [B, 1, 3] - the selected pt

        avail_i = tf.squeeze(tf.map_fn(tf.where, select_m, dtype=tf.int64), axis=-1)  # [B, N'] - NOTE each batch should have different avail points (?)
        avail_p = tf.gather(points, avail_i, batch_dims=1)  # [B, N' 3]  - avail points
        avail_d = tf.gather(dist_N, avail_i, batch_dims=1)  # [B, N']    - current dist of avail points
        avail_I = tf.gather(dist_i, avail_i)  # [B, N']    - idx corresponding to avail points

        update_d = tf.sqrt(tf.reduce_sum((update_p - avail_p)** 2, axis=-1))  # [B, N'] - dist to current selected pt
        update_d = tf.math.minimum(avail_d, update_d)  # [B, N']  - new dist

        next_i = tf.gather(avail_I, tf.expand_dims(tf.argmax(update_d, axis=-1), axis=-1), batch_dims=1)  # [B, 1]

        select = tf.concat([select, next_i], axis=-1)  # [B, idx + 1] - update select
        dist_N = tf.tensor_scatter_update(dist_N, avail_i, update_d)  # update dist_N NOTE: scatter update with batch???
        raise
        select_m = tf.tensor_scatter_update(select_m, next_i, select_f)  # update select_m
        idx += 1
        return idx, dist_N, select, select_m


    def body_mask(idx, dist_N, select):  # idx = 0 -> n
        update_i = select[..., -1:]  # [B, 1] - the selected idx
        # update_i = tf.gather(select, [idx - 1], axis=1)
        update_p = tf.gather(points, update_i, batch_dims=1)  # [B, 1, 3] - the selected pt
        update_d = tf.reduce_sum((update_p - points) ** 2, axis=-1)  # [B, N] - dist^2 to current selected pt

        # idx = tf.Print(idx, ['idx', idx, 'update_i/p/d', update_i, update_p, update_d, 'dist_N', dist_N, 'select', select], summarize=10 ** 3)
        dist_N = tf.math.minimum(dist_N, update_d)  # [B, N']  - new dist

        next_i = tf.expand_dims(tf.argmax(dist_N, axis=-1), axis=-1)  # [B, 1] - dist of the selected pt = 0

        select = tf.concat([select, next_i], axis=-1)  # [B, idx + 1] - update select
        # select_m = tf.map_fn(lambda x: tf.tensor_scatter_update(*x), (select_m, tf.expand_dims(next_i, axis=-1), select_f), dtype=tf.bool)  # update select_m
        idx += 1
        return idx, dist_N, select

    def cond(idx, dist_N, select):
        return tf.less(idx, n)

    rst = tf.while_loop(cond=cond, body=body_mask,
                        loop_vars=[1, dist_N, select],
                        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None, None])])
    idx, dist_N, select = rst
    # select = tf.Print(select, ['idx', n, 'dist_N', dist_N, 'select', select], summarize=10 ** 3)
    return select


# import numba
# from numba import jit

def _distance(a, b):
    a_min_b = a - b
    return np.einsum("ij,ij->i", a_min_b, a_min_b)

def farthest_sample_np(points, n, batch_inds=None):
    rank = len(points.shape)
    if batch_inds is not None:
        raise
    if rank == 3:  # [B, N, 3]
        return np.vectorize(partial(farthest_sample_np, n=n), points)

    # farthest sampling on [N, 3]
    assert rank == 2

    update_i = np.zeros(n)
    distance = np.zeros(points.shape[0]) + np.inf  # [N] - dist of points to sampled points set
    for i in range(1, n):
        update_p = points[update_i[i-1]]  # [3] - current selected points (starts from points[0])
        # update_d = ((points - update_p) ** 2).sum(aixs=-1)  # [N] - distance of all points to current selection
        update_d = _distance(points, update_p)
        distance = np.minimum(distance, update_d)  # update distance
        update_i[i] = distance.argmax()
    return update_i

