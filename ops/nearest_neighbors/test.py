import sys, time
import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors

s = int(sys.argv[1]) if len(sys.argv) > 1 else 123456
np.random.seed(s)

batch_size = 1
num_points = 20
K = 5
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours
start = time.time()
neigh_idx = nearest_neighbors.knn_batch(pc, pc, K, omp=True)  # return 0 if K > num_points
print(time.time() - start)

print(neigh_idx.shape)
print(neigh_idx)
