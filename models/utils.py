import os, re
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

from utils.ply import read_ply, write_ply


def kernel_point_optimization_debug(radius, num_points,
                                    num_kernels=1, dimension=3,
                                    fixed='center', ratio=1.0, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initiate figure
    # if verbose>1:
    #     fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        # if verbose > 1:
        #     plt.clf()
        #     plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
        #     circle = plt.Circle((0, 0), radius, color='r', fill=False)
        #     fig.axes[0].add_artist(circle)
        #     fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
        #     fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
        #     fig.axes[0].set_aspect('equal')
        #     plt.draw()
        #     plt.pause(0.001)
        #     plt.show(block=False)
        #     print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def create_kernel_points(scope, radius, num_kpoints, num_kernels, dimension, fixed, config):
    # Number of tries in the optimization process, to ensure we get the most stable disposition
    num_tries = 100

    # Kernel directory
    kernel_dir = os.path.join(config.saving_path, 'kernels')
    if not os.path.exists(kernel_dir):
        os.makedirs(kernel_dir)

    prefix_name = scope.name.split('Model/')[-1].replace('/', '_')

    if dimension == 3:
        specific_kernel_file = os.path.join(kernel_dir,
                                            'sk_{}_{:04f}_{:03d}_{:s}.npy'.format(prefix_name, radius, num_kpoints,
                                                                                  fixed))
    elif dimension == 2:
        specific_kernel_file = os.path.join(kernel_dir,
                                            'sk_{}_{:04f}_{:03d}_{:s}_2D.npy'.format(prefix_name, radius, num_kpoints,
                                                                                     fixed))
    else:
        raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))

    if os.path.exists(specific_kernel_file):
        kernels = np.load(specific_kernel_file)
    else:
        # Kernel_file
        if dimension == 3:
            kernel_file = os.path.join(kernel_dir, 'k_{:03d}_{:s}.ply'.format(num_kpoints, fixed))
        elif dimension == 2:
            kernel_file = os.path.join(kernel_dir, 'k_{:03d}_{:s}_2D.ply'.format(num_kpoints, fixed))
        else:
            raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))

        # Check if already done
        if not os.path.exists(kernel_file):

            # Create kernels
            kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                        num_kpoints,
                                                                        num_kernels=num_tries,
                                                                        dimension=dimension,
                                                                        fixed=fixed,
                                                                        verbose=0)

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            original_kernel = kernel_points[best_k, :, :]
            write_ply(kernel_file, original_kernel, ['x', 'y', 'z'])

        else:
            data = read_ply(kernel_file)
            original_kernel = np.vstack((data['x'], data['y'], data['z'])).T

        # N.B. 2D kernels are not supported yet
        if dimension == 2:
            return original_kernel

        # Random rotations depending of the fixed points
        if fixed == 'verticals':

            # Create random rotations
            thetas = np.random.rand(num_kernels) * 2 * np.pi
            c, s = np.cos(thetas), np.sin(thetas)
            R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
            R[:, 0, 0] = c
            R[:, 1, 1] = c
            R[:, 2, 2] = 1
            R[:, 0, 1] = s
            R[:, 1, 0] = -s

            # Scale kernels
            original_kernel = radius * np.expand_dims(original_kernel, 0)

            # Rotate kernels
            kernels = np.matmul(original_kernel, R)

        else:

            # Create random rotations
            u = np.ones((num_kernels, 3))
            v = np.ones((num_kernels, 3))
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
            while np.any(wrongs):
                new_u = np.random.rand(num_kernels, 3) * 2 - 1
                new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
                u[wrongs, :] = new_u[wrongs, :]
                new_v = np.random.rand(num_kernels, 3) * 2 - 1
                new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
                v[wrongs, :] = new_v[wrongs, :]
                wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

            # Make v perpendicular to u
            v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
            v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

            # Last rotation vector
            w = np.cross(u, v)
            R = np.stack((u, v, w), axis=-1)

            # Scale kernels
            original_kernel = radius * np.expand_dims(original_kernel, 0)

            # Rotate kernels
            kernels = np.matmul(original_kernel, R)

            # Add a small noise
            kernels = kernels
            kernels = kernels + np.random.normal(scale=radius * 0.01, size=kernels.shape)

        np.save(specific_kernel_file, kernels)

    return kernels


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return tf.exp(-sq_r / (2 * tf.square(sig) + eps))


# ---------------------------------------------------------------------------- #
# decorator
# ---------------------------------------------------------------------------- #

def tf_scope(func):
    """ decorator: automatically wrap a var scope """
    def scopped_func(*args, name=None, reuse=None, **kwargs):
        if name is not None and not reuse:
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        elif name is not None and reuse:  # variable reuse, naming ops as desired
            with tf.variable_scope(reuse, auxiliary_name_scope=False, reuse=True):
                with tf.name_scope(name):
                    return func(*args, **kwargs)
        elif reuse:  # variable reuse + naming ops as is re-enter the scope
            with tf.variable_scope(reuse, reuse=True):
                    return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return scopped_func

def tf_device(func=None, default=None):
    # print(func, default)

    def tf_device_with_default(func):
        """ decorator: automatically wrap a device scope """
        def scopped_func(*args, device=default, **kwargs):
            if device is not None:
                with tf.device(device):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return scopped_func

    if default is not None:
        # return a new decorator with `device=default` specified
        return tf_device_with_default

    assert func is not None, f'called as decorator, but received func = {func}'
    """ decorator: automatically wrap a device scope """
    def scopped_func(*args, device=None, **kwargs):
        if device is not None:
            with tf.device(device):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    # directly wrap the func
    return scopped_func


def tf_Print(*args, summarize=100, **kwargs):
    if 'summarize' in kwargs:
        summarize = kwargs['summarize']
        del kwargs['summarize']
    with tf.device('/cpu:0'): return tf.Print(*args, summarize=summarize, **kwargs)


# ---------------------------------------------------------------------------- #
# helper func
# ---------------------------------------------------------------------------- #

def get_kr(config, stage_n, stage_i):
    assert stage_n in _valid_stage, f'invalid stage_n={stage_n}'
    if stage_n:
        kr = config.kr_sample[stage_i - 1] if stage_n == 'down' else config.kr_sample_up[stage_i]
    else:
        kr = config.kr_search[stage_i]
    return kr

def get_kwargs(block_cfg, config, is_training, act=False):
    # NOTE: may consider provide bias, bn, activation - 1. matching def of dense_layer & mlps, 2. arch level bn control, e.g. ln/gn
    kwargs = {
        'is_training': is_training,
        'initializer': block_cfg.init if block_cfg.init != '' else config.init,
        'weight_decay': block_cfg.wd if block_cfg.wd != '' else config.weight_decay,
        'bn_momentum': config.bn_momentum, 'bn_eps': config.bn_eps,
    }
    if block_cfg.bn != '' or config.bn != '':
        kwargs['bn'] = block_cfg.bn if block_cfg.bn != '' else config.bn
    if act is True:
        kwargs['activation'] = block_cfg.act if block_cfg.act else config.activation
    elif act:
        kwargs['activation'] = act
    return kwargs

def get_kwargs_mlp(block_cfg, config, is_training, act=True, **_kwargs):
    kwargs = get_kwargs(block_cfg, config, is_training, act=act)
    kwargs.update({
        'linearbn': block_cfg.linearbn if block_cfg.linearbn != '' else config.linearbn if config.linearbn != '' else False,
    })
    kwargs.update(_kwargs)
    return kwargs

def get_ftype(ftype, raise_not_found=True):
    if ftype in ['out', 'fout', 'f_out']:
        ptype = 'p_out'
        ftype = 'f_out'
    elif any(re.fullmatch(f'{k}(\d*mlp|mlp\d*|linear|)', ftype) for k in ['latent', 'logits', 'probs']):
        ptype = 'p_out'
        ftype = [k for k in ['latent', 'logits', 'probs'] if ftype.startswith(k)][0]
    elif ftype in ['sample', 'fsample', 'f_sample']:
        ptype = 'p_sample'
        ftype = 'f_sample' if ftype in ['sample', 'fsample'] else ftype
    elif raise_not_found:
        raise KeyError(f'not supported ftype = {ftype}')
    else:
        ftype = ptype = None
    return ftype, ptype


_valid_stage = ['down', 'up', '']
def fetch_supports_flow(inputs, stage_n, stage_i):
    # update based on the flowing direction down/up - building
    assert stage_n in _valid_stage, f'invalid stage_n={stage_n}'
    if stage_n:
        stage_i += -1 if stage_n == 'down' else 1
        idx = inputs['sample_idx'][stage_n][stage_i]
        pts = inputs['points'][stage_i]
    else:
        idx = inputs['neighbors'][stage_i]
        pts = inputs['points'][stage_i]
    return pts, idx

def fetch_supports_stage(inputs, stage_n, stage_i, ftype):
    # indexing the existing stages - all built
    stage_n = to_valid_stage(stage_n)
    stage = inputs['stage_list'][stage_n][stage_i]
    ftype, ptype = get_ftype(ftype)
    pts = stage[ptype]
    f = stage[ftype]
    idx = inputs['neighbors'][stage_i]
    return pts, f, idx

def to_valid_stage(stage_n, short=False):
    if stage_n in ['D', 'down']:
        stage_n = 'D' if short else 'down'
    elif stage_n in ['U', 'up']:
        stage_n = 'U' if short else 'up'
    elif stage_n in ['S', 'stages']:
        stage_n = 'S' if short else 'stages'
    else:
        raise ValueError(f'invalid stage_n={stage_n}')
    return stage_n

def parse_stage(stage, num_layers):
    stage_list = [i.strip('_') for i in re.split('([A-Z])', stage) if i and i.strip('_')]  # e.g. D012_U34
    assert len(stage_list) % 2 == 0, f'invalid stage compound: stage_list={stage_list} from stage={stage}'
    stage_n = [s for i, s in enumerate(stage_list) if i % 2 == 0]
    stage_i = [s for i, s in enumerate(stage_list) if i % 2 == 1]
    # stage_list = [[(to_valid_stage(n), int(i)) for i in i_str] for n, i_str in zip(stage_n, stage_i)]
    # stage_list = sum(stage_list, [])
    stage_list = []
    for n, i_str in zip(stage_n, stage_i):
        if i_str.startswith('a'):
            i_list = [i for i in range(num_layers) if f'{i}' not in i_str]  # Ua0 - all except 0
        else:
            i_list = [int(i) for i in i_str]
        n = to_valid_stage(n)
        stage_list += [(n, i) for i in i_list]
    return stage_list

def get_batch_inds(inputs, stage_i):
    if inputs['batches_ind'][stage_i] is None:
        inputs['batches_ind'][stage_i] = tf_get_batch_inds(inputs['batches_len'][stage_i])
    return inputs['batches_ind'][stage_i]

def tf_get_batch_inds(stacks_len):
    """Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
    From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    """

    # Initiate batch inds tensor
    num_batches = tf.shape(stacks_len)[0]
    num_points = tf.reduce_sum(stacks_len)
    batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

    # Define body of the while loop
    def body(batch_i, point_i, b_inds):
        num_in = stacks_len[batch_i]
        num_before = tf.cond(tf.less(batch_i, 1),
                                lambda: tf.zeros((), dtype=tf.int32),
                                lambda: tf.reduce_sum(stacks_len[:batch_i]))
        num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                            lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                            lambda: tf.zeros((), dtype=tf.int32))

        # Update current element indices
        inds_before = tf.zeros((num_before,), dtype=tf.int32)
        inds_in = tf.fill((num_in,), batch_i)
        inds_after = tf.zeros((num_after,), dtype=tf.int32)
        n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

        b_inds += n_inds

        # Update indices
        point_i += stacks_len[batch_i]
        batch_i += 1

        return batch_i, point_i, b_inds

    def cond(batch_i, point_i, b_inds):
        return tf.less(batch_i, tf.shape(stacks_len)[0])

    _, _, batch_inds = tf.while_loop(cond, body,
        loop_vars=[0, 0, batch_inds_0], shape_invariants=[tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None])])

    return batch_inds

def stack_batch_inds(inputs, stage_i):
    if inputs['batches_stack'][stage_i] is None:
        inputs['batches_stack'][stage_i] = tf_stack_batch_inds(inputs['batches_len'][stage_i])
    return inputs['batches_stack'][stage_i]

def tf_stack_batch_inds(*args, impl='tfwhile', **kwargs):
    impl = impl[2:] if impl.startswith('tf') else impl
    func = {
        'map': tf_stack_batch_inds_map,
        'while': tf_stack_batch_inds_while,
    }[impl]
    return func(*args, **kwargs)

def tf_stack_batch_inds_while(stacks_len, num_points=None, tight=True):
    """
    Stack the flat point idx, given the batch element sizes (stacks_len)
        E.g. stacks_len = [3, 2, 5]; n = sum(stacks_len) = 10
        => return: [[0, 1, 2, n, n, n], 
                    [3, 4, n, n, n, n],
                    [5, 6, 7, 8, 9, n]]
    """
    # Initiate batch inds tensor
    num_points = num_points if num_points is not None else tf.reduce_sum(stacks_len)
    max_points = tf.reduce_max(stacks_len)
    batch_inds_0 = tf.zeros((0, max_points), dtype=tf.int32)

    # Define body of the while loop
    def body(batch_i, point_i, b_inds):
        # Create this element indices
        element_inds = tf.expand_dims(tf.range(point_i, point_i + stacks_len[batch_i]), axis=0)
        # Pad to right size
        padded_inds = tf.pad(element_inds,
                                [[0, 0], [0, max_points - stacks_len[batch_i]]],
                                "CONSTANT",
                                constant_values=num_points)
        # Concatenate batch indices
        b_inds = tf.concat((b_inds, padded_inds), axis=0)
        # Update indices
        point_i += stacks_len[batch_i]
        batch_i += 1
        return batch_i, point_i, b_inds

    def cond(batch_i, point_i, b_inds):
        return tf.less(batch_i, tf.shape(stacks_len)[0])

    fixed_shapes = [tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, None])]
    _, _, batch_inds = tf.while_loop(cond, body, loop_vars=[0, 0, batch_inds_0], shape_invariants=fixed_shapes)

    # Add a last column with shadow neighbor if there is not
    def f1(): return tf.pad(batch_inds, [[0, 0], [0, 1]], "CONSTANT", constant_values=num_points)
    def f2(): return batch_inds
    if not tight:
        batch_inds = tf.cond(tf.equal(num_points, max_points * tf.shape(stacks_len)[0]), true_fn=f1, false_fn=f2)

    return batch_inds

def tf_stack_batch_inds_map(stacks_len, num_points=None, tight=True):
    # Initiate batch inds tensor
    B_inds = tf.range(tf.shape(stacks_len)[0])  # [B]
    num_points = num_points if num_points is not None else tf.reduce_sum(stacks_len)
    max_points = tf.reduce_max(stacks_len)
    if not tight:
        max_points += 1
    def flatten_idx(batch_i):
        cur_len = stacks_len[batch_i]
        start_i = tf.reduce_sum(stacks_len[:batch_i])
        element_inds = tf.range(start_i, start_i + cur_len)  # Create this element indices (starting at 0)
        padded_inds = tf.pad(element_inds, [[0, max_points - cur_len]], "CONSTANT", constant_values=num_points)  # [max_points] Pad to right size
        return padded_inds
    batch_inds = tf.map_fn(flatten_idx, B_inds, dtype=tf.int32)
    return batch_inds

