import sys, time
import numpy as np

import tensorflow as tf
from tf_knnquery import knnquery_cuda

s = int(sys.argv[1]) if len(sys.argv) > 1 else 123456
np.random.seed(s)

batch_size = 1
num_points = 20
K = 5
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours
neigh_idx = knnquery_cuda(pc, pc, None, None, K)
print(neigh_idx.shape)
print(neigh_idx)

s = tf.Session()

start = time.time()
idx = s.run(neigh_idx)
print(time.time() - start)
print(idx)
