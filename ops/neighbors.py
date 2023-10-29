import numpy as np
import numba
from numba import jit

@jit(nopython=True, parallel=True)
def calc_dist(xyz, points):  # squared
    return ((xyz - points) ** 2).sum(axis=-1)

@jit(nopython=True, parallel=True)
def get_dist_idx(xyz, points, radius):
    dist = calc_dist(xyz, points)  # [BxN]
    dist_idx = dist.argsort()
    return dist_idx, (dist < radius).sum()

# @jit
def radius_neighbors_clip(queries, supports, radius, q_start, s_start, neighbors):
    radius = radius ** 2
    max_neighbor = neighbors.shape[1]
    for i in range(len(queries)):
        xyz = queries[i]
        dist_idx, dist_num = get_dist_idx(xyz, supports, radius)
        num = min(dist_num, max_neighbor)
        neighbors[q_start + i][:num] = s_start + dist_idx[:num]
        if i % 1000 == 0:
            print(f'{i}/{len(queries)}')
    return neighbors

# @jit
def radius_batch_neighbors_clip(queries, supports, q_batches, s_batches, radius, max_neighbor=50):
    q_start = s_start = 0
    start_list = []
    for q_len, s_len in zip(q_batches, s_batches):
        start_list += [(q_start, s_start)]
        q_start += q_len
        s_start += s_len

    neighbors = np.zeros((queries.shape[0], max_neighbor)) + len(supports)
    iter = numba.prange(len(start_list)) if len(start_list) > 1 else range(len(start_list))  # parallel threshold
    for i in iter:
        q_start, s_start = start_list[i]
        q_len = q_batches[i]
        s_len = s_batches[i]
        neighbors = radius_neighbors_clip(queries[q_start:q_start+q_len], supports[s_start:s_start+s_len], radius, q_start, s_start, neighbors)

    # neighbors += np.array(neighbors < 0, dtype=float) * (len(supports) + 1)
    # idx = np.where(neighbors < 0)
    # for i, j in zip(idx):
    #     neighbors[i, j] = len(supports)
    return neighbors

def radius_neighbors(queries, supports, radius):
    radius = radius ** 2
    idx = np.arange(len(supports))
    neighbor_idx = []
    for i, xyz in enumerate(queries):
        dist = calc_dist(xyz, supports)  # [BxN]
        idx = np.where(dist < radius)
        neighbor_idx.append(idx)
    return neighbor_idx

def radius_batch_neighbors(queries, supports, q_batches, s_batches, radius):

    q_start = s_start = 0
    start_list = []
    for q_len, s_len in zip(q_batches, s_batches):
        start_list += [(q_start, s_start)]
        q_start += q_len
        s_start += s_len

    neighbor_list = [None] * len(start_list)
    iter = numba.prange(len(start_list)) if len(start_list) > 1 else range(len(start_list))  # parallel threshold
    for i in iter:
        q_start, s_start = start_list[i]
        q_len = q_batches[i]
        s_len = s_batches[i]
        neighbor_idx = radius_neighbors(queries[q_start:q_start+q_len], supports[s_start:s_start+s_len], radius)
        neighbor_list[i] = neighbor_idx

    neighbor_list = sum(neighbor_list, [])
    max_len = max(len(n) for n in neighbor_list)
    for i, n in enumerate(neighbor_list):
        if len(n) == max_len:
            continue
        fill = np.zeros([max_len - len(n)]) - 1
        neighbor_list[i] = np.concatenate(n, fill) if n else fill
    neighbors = np.array(neighbor_list)
    neighbors[neighbors < 0] = len(supports)
    return neighbors