import os, sys
import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_knn.so'))

def knnquery_cuda(queries, supports, q_batches, s_batches, k):
    '''
    input:
        queries     : [BxNq, 3]
        supports    : [BxNs, 3]
        q_batches   : [B]/None
        s_batches   : [B]/None
        k           : int
    returns:
        neighbor_idx: [BxNq, k]
    '''
    B = None
    q_shape = tf.shape(queries)

    if q_batches is None:  # [B, Nq, 3] -> [BNq, 3]
        B = q_shape[0]
        queries = tf.reshape(queries, [-1, 3])
        q_batches = tf.tile([q_shape[1]], multiples=[B])
    if s_batches is None:  # [B, Ns, 3] -> [BNs, 3]
        s_shape = tf.shape(supports)
        B = B if B is not None else s_shape[0]
        supports = tf.reshape(supports, [-1, 3])
        s_batches = tf.tile([s_shape[1]], multiples=[B])

    # batches -> offset - cumsum not support int32 on gpu...???
    q_offset = tf.cast(tf.cumsum(tf.cast(q_batches, tf.float32)), tf.int32)
    s_offset = tf.cast(tf.cumsum(tf.cast(s_batches, tf.float32)), tf.int32)

    idx = module.knn_query(queries, supports, q_offset, s_offset, k)  # [BNq, k]
    idx = tf.reshape(idx, tf.concat([q_shape[:-1], [k]], axis=0))  # [BxNq, k]
    # idx = tf.stop_gradient(idx)
    return idx
ops.NoGradient('KnnQuery')
