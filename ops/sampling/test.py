import time
import numpy as np
import tensorflow as tf
from tf_sampling import prob_sample, gather_point, farthest_point_sample

sess = tf.InteractiveSession()

np.random.seed(100)
triangles = np.random.rand(1, 5, 3, 3).astype('float32')

inp = tf.constant(triangles)
tria = inp[:,:, 0,:]
trib = inp[:,:, 1,:]
tric = inp[:,:, 2,:]
areas = tf.sqrt(tf.reduce_sum(tf.cross(trib - tria, tric - tria)** 2, 2) + 1e-9)

print(triangles)
print('areas =', areas)

rand_num = tf.constant(np.random.uniform(size=(1, 8192)), dtype=tf.float32)
triids = prob_sample(areas, rand_num)

tria_sample = gather_point(tria, triids)
trib_sample = gather_point(trib, triids)
tric_sample = gather_point(tric, triids)

us = tf.constant(np.random.uniform(size=(1, 8192)), dtype=tf.float32)
vs = tf.constant(np.random.uniform(size=(1, 8192)), dtype=tf.float32)

uplusv = 1 - tf.abs(us + vs - 1)
uminusv = us - vs
us = (uplusv + uminusv) * 0.5
vs = (uplusv - uminusv) * 0.5

pt_sample = tria_sample + (trib_sample - tria_sample) * tf.expand_dims(us, -1) + (tric_sample - tria_sample) * tf.expand_dims(vs, -1)
print('pt_sample:', pt_sample)

reduced_sample = gather_point(pt_sample, farthest_point_sample(1024, pt_sample))
print('reduced_sample:', reduced_sample)

start = time.time()
ret = sess.run([pt_sample, reduced_sample])
print(time.time() - start); start = time.time()

pt, ret = ret
print(pt.shape, pt.dtype)
print(pt)
print('=' * 30)
print(ret.shape, ret.dtype)
print(ret)

from farthest_sample import farthest_sample
n = 1024
# g = farthest_sample(pt_sample, n, body='gather')
with tf.device('/cpu:0'):
    m_i = farthest_sample(pt_sample, n)
    m_p = tf.gather(pt_sample, m_i, batch_dims=1)

# print('gather')
# start=time.time()
# ret = sess.run(g)
# print(time.time() - start); start = time.time()
# print(ret)


print('mask')
start = time.time()
ret = sess.run([m_i, m_p])
print('timing = ', time.time() - start); start = time.time()
print(ret[1].shape, ret[1].dtype)
print(ret[0])
print(ret[1])
