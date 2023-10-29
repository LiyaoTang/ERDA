import os, re, sys
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

OPS_DIR = os.path.join(ROOT_DIR, 'ops')
sys.path.insert(0, OPS_DIR)

class Dataset(object):
    def __init__(self, config):
        self.config = config

        # path
        self.data_path = config.data_path if config.data_path else 'Data'
        self.data_path = f'{self.data_path}/{config.dataset}'
        assert os.path.exists(self.data_path), f'invalid data_path = {self.data_path}'

        # interface - init op
        self.train_init_op = None
        self.val_init_op = None
        self.test_init_op = None

    @property
    def info(self):
        info = {
            'ignored_labels': self.ignored_labels,
            'label_names': self.label_names,
        }
        if self.config.ignore_list:
            info['ignored_labels'] = list(info['ignored_labels'])
        # if hasattr(self, 'ignored_labels_test'):
        #     info.update({'ignored_labels_test': self.ignored_labels_test})
        return info

    def valid_split(self, split, short=False):
        assert split in ['train', 'training', 'val', 'validation', 'test'], f'invalid split = {split}'
        if split.startswith('train'):
            return 'train' if short else 'training'
        elif split.startswith('val'):
            return 'val' if short else 'validation'
        else:
            return 'test'

    def init_labels(self):
        """
        Initiate all label parameters given the label_to_names dict
        """
        self.num_classes = len(self.label_to_names) - len(self.ignored_labels)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # may not be consecutive or start from 0
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        # original label value <-> idx of valid label
        if not hasattr(self, 'label_to_idx'):
            self.label_to_idx = []
            idx = 0
            for l in self.label_values:
                while len(self.label_to_idx) < l:
                    self.label_to_idx += [None]  # skipped labels - not even invalid i.e. should not exists in label idx
                self.label_to_idx += [idx] if l not in self.ignored_labels else [-1]
                idx += l not in self.ignored_labels
            self.label_to_idx = np.array(self.label_to_idx)

        if not hasattr(self, 'idx_to_label'):
            self.idx_to_label = np.array([l for l in self.label_values if l not in self.ignored_labels])

        # # full label -> subset label (those in label_to_names)
        # self.reduced_labels = np.array([self.idx_to_label[i] if i is not None and i > 0 else None for i in self.label_to_idx])

        if self.config.cloud_labels == 'multi':
            self.num_classes = int(sum(self.cloud_labels_multi))
            self.label_to_idx = self.idx_to_label = np.arange(self.num_classes)
            assert not len(self.ignored_labels)
        assert self.config.num_classes == self.num_classes

    def initialize(self, verbose=True):
        config = self.config
        # initialize op
        if config.search == 'radius':
            self.initialize_radius(verbose=verbose)
        elif config.search in ['knn', 'knn_gpu']:
            self.initialize_fixed_size(verbose=verbose)
        else:
            raise NotImplementedError(f'not supported methods: sampling = {config.sample}; searching = {config.search}')

    def initialize_radius(self, verbose=True):
        config = self.config
        self.batch_limit = self.calibrate_batches('training', config.batch_size)  # max num points [BxN] of a batch - used in get_batch_gen
        self.batch_limit_val = self.calibrate_batches('validation', config.batch_size_val) if config.batch_size_val else None
        # neighbor_limits - used in base.big_neighborhood_filter => set neighbor_idx shape
        self.neighborhood_limits = config.neighborhood_limits if config.neighborhood_limits else self.calibrate_neighbors('training')
        if config.max_neighborhood_limits:
            self.neighborhood_limits = [min(i, config.max_neighborhood_limits) for i in self.neighborhood_limits]
        self.neighborhood_limits = [int(l * config.density_parameter // 5) for l in self.neighborhood_limits]
        if verbose:
            print("batch_limit: ", self.batch_limit)
            print("neighborhood_limits: ", self.neighborhood_limits)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_radius('training')
        gen_function_val, _, _ = self.get_batch_gen_radius('validation')
        gen_function_test, _, _ = self.get_batch_gen_radius('test')
        kwargs = gen_function.kwargs if hasattr(gen_function, 'kwargs') else {}
        map_func = self.get_tf_mapping_radius(**kwargs)

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        # self.train_data = self.train_data.apply(tf.data.experimental.copy_to_device('/gpu:0'))
        self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE)
        # self.train_data = self.train_data.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_data = None
        if gen_function_test is not None:
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            self.test_data = self.test_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
            self.test_data = self.test_data.prefetch(tf.data.experimental.AUTOTUNE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        # independent stream for each gpus
        self.flat_inputs = [iter.get_next() for i in range(config.gpu_num)]
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.test_init_op = iter.make_initializer(self.test_data) if self.test_data is not None else None

    def initialize_fixed_size(self, verbose=True):
        config = self.config
        if verbose:
            print('\n\t'.join(['k-nn & ratio:'] + [f'{a} = {getattr(config, a)}' for a in ['kr_search', 'kr_sample', 'kr_sample_up', 'r_sample']]))

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_fixed_size('training')
        gen_function_val, _, _ = self.get_batch_gen_fixed_size('validation')
        gen_function_test, _, _ = self.get_batch_gen_fixed_size('test')
        map_func = self.get_tf_mapping_fixed_size()

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.batch(config.batch_size)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.batch(config.batch_size_val if config.batch_size_val else config.batch_size)
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_data = None
        if gen_function_test is not None:
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            self.test_data = self.test_data.batch(config.batch_size_val if config.batch_size_val else config.batch_size)
            self.test_data = self.test_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
            self.test_data = self.test_data.prefetch(tf.data.experimental.AUTOTUNE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        # independent stream for each gpus
        self.flat_inputs = [iter.get_next() for i in range(config.gpu_num)]
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.test_init_op = iter.make_initializer(self.test_data) if self.test_data is not None else None


    def calibrate_batches(self, split=None, batch_size=None):
        s = 'training' if len(self.input_trees['training']) > 0 else 'test'
        split = split if split else s
        batch_size = batch_size if batch_size else self.config.batch_size

        N = (10000 // len(self.input_trees[split])) + 1
        sizes = []
        # Take a bunch of example neighborhoods in all clouds
        for i, tree in enumerate(self.input_trees[split]):
            # Randomly pick points
            points = np.array(tree.data, copy=False)
            rand_inds = np.random.choice(points.shape[0], size=N, replace=False)
            rand_points = points[rand_inds]
            noise = np.random.normal(scale=self.config.in_radius / 4, size=rand_points.shape)
            rand_points += noise.astype(rand_points.dtype)
            neighbors = tree.query_radius(points[rand_inds], r=self.config.in_radius)
            # Only save neighbors lengths
            sizes += [len(neighb) for neighb in neighbors]
        sizes = np.sort(sizes)
        # Higher bound for batch limit
        lim = sizes[-1] * batch_size
        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            if sum_s > lim:
                max_b = i
                break
        # With a proportional corrector, find batch limit which gets the wanted batch_num
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False)
            b = np.sum(np.cumsum(rand_shapes) < lim)
            # Update estim_b (low pass filter istead of real mean
            estim_b += (b - estim_b) / min(i + 1, 100)
            # Correct batch limit
            lim += 10.0 * (batch_size - estim_b)
        return lim

    def calibrate_neighbors(self, split, keep_ratio=0.8, samples_threshold=10000):

        # Create a tensorflow input pipeline
        # **********************************
        import time
        config = self.config
        assert split in ['training', 'test']

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)

        # Init batch limit if not done
        self.batch_limit = self.batch_limit if hasattr(self, 'batch_limit') else self.calibrate_batches()

        # Get mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_radius(split)
        kwargs = gen_function.kwargs if hasattr(gen_function, 'kwargs') else {}
        map_func = self.get_tf_mapping_radius(**kwargs)

        # Create batched dataset from generator
        train_data = tf.data.Dataset.from_generator(gen_function,
                                                    gen_types,
                                                    gen_shapes)

        train_data = train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        train_data = train_data.prefetch(10)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        flat_inputs = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_data)

        # Create a local session for the calibration.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        with tf.Session(config=cProto) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Initialise iterator with train data
            sess.run(train_init_op)

            # Get histogram of neighborhood sizes in 1 epoch max
            # **************************************************

            neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
            t0 = time.time()
            mean_dt = np.zeros(2)
            last_display = t0
            epoch = 0
            training_step = 0
            while epoch < 1 and np.min(np.sum(neighb_hists, axis=1)) < samples_threshold:
                try:

                    # Get next inputs
                    t = [time.time()]
                    ops = flat_inputs['neighbors']
                    neighbors = sess.run(ops)
                    t += [time.time()]

                    # Update histogram
                    counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    t += [time.time()]

                    # Average timing
                    mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display
                    if (t[-1] - last_display) > 2.0:
                        last_display = t[-1]
                        message = 'Calib Neighbors {:08d} : timings {:4.2f} {:4.2f}'
                        print(message.format(training_step, 1000 * mean_dt[0], 1000 * mean_dt[1]))

                    training_step += 1

                except tf.errors.OutOfRangeError:
                    print('End of train dataset')
                    epoch += 1

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

            self.neighborhood_limits = percentiles
            print('neighborhood_limits : {}'.format(self.neighborhood_limits))

        return


    def init_sampling(self, split):
        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':  # First compute the number of point we want to pick in each cloud set - num of samples
            epoch_n = self.config.epoch_steps * self.config.epoch_batch
        elif split == 'validation':
            epoch_n = self.config.validation_steps * self.config.epoch_batch
        elif split == 'test':
            epoch_n = self.config.validation_steps * self.config.epoch_batch
        elif split == 'ERF':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = 1000000
            self.batch_limit = 1  # BxN = 1, single point
            np.random.seed(42)
            split = 'test'
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        for i, tree in enumerate(self.input_trees[split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]
        self.min_potentials[split] = np.array(self.min_potentials[split])

        if self.min_potentials[split].size == 0:
            self.min_potentials[split] = np.random.rand(len(self.input_names[split])) * 1e-3

        return epoch_n

    def get_batch_gen_radius(self, split):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        config = self.config
        epoch_n = self.init_sampling(split)
        data_split = split
        batch_limit = self.batch_limit
        if split != 'training' and self.batch_limit_val:
            batch_limit = self.batch_limit_val
        rgb_dims = self.rgb_dims if hasattr(self, 'rgb_dims') else 3

        ##########################
        # Def generators functions
        ##########################
        def spatially_regular_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                if split != 'ERF':
                    noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                    pick_point = center_point + noise.astype(center_point.dtype)
                else:
                    pick_point = center_point

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                                  r=config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                if split != 'ERF':
                    dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(config.in_radius))
                    tukeys[dists > np.square(config.in_radius)] = 0
                    self.potentials[split][cloud_ind][input_inds] += tukeys
                    self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                    # Safe check for very dense areas - align with training setting
                    if n > self.batch_limit:
                        input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                        n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split in ['test', 'ERF']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]
                    # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),      # [BxN, 3]  - xyz in sample
                           np.concatenate(c_list, axis=0),      # [BxN, 3/1 + 3 (RGB/intensity + global xyz in whole cloud)]
                           np.concatenate(pl_list, axis=0),     # [BxN]     - labels
                           np.array([tp.shape[0] for tp in p_list]),    # [B]    - size (point num) of each batch
                           np.concatenate(pi_list, axis=0),             # [B, N] - point idx in each of its point cloud
                           np.array(ci_list, dtype=np.int32))           # [B]    - cloud idx

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))
        spatially_regular_gen.types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        spatially_regular_gen.shapes = ([None, 3], [None, 3 + rgb_dims], [None], [None], [None], [None])

        def spatially_regular_weaksup_gen():
            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            batch_n = 0
            # Generator loop
            i = 0
            while i < epoch_n:
                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))
                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])
                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)
                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point, r=config.in_radius)[0]
                # Number collected
                n = input_inds.shape[0]
                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(config.in_radius))
                tukeys[dists > np.square(config.in_radius)] = 0
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                if split == 'training' and not any(i.startswith('pseudo') for i in config.architecture):
                    label_inds = np.where(~np.isin(self.input_labels[split][cloud_ind], self.ignored_labels))[0]  # all valid labels
                    input_label_mask = ~np.isin(self.input_labels[data_split][cloud_ind][input_inds], self.ignored_labels)
                    input_label_inds = input_inds[input_label_mask]  # current valid labels
                    other_label_mask = ~np.isin(label_inds, input_label_inds)
                    label_inds = label_inds[other_label_mask]  # concat other valid labels
                    input_inds = np.concatenate([label_inds, input_inds])
                    n = input_inds.shape[0]

                # Safe check for very dense areas - align with training setting
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                    n = input_inds.shape[0]

                # Get label
                if split in ['test', 'ERF']:
                    input_labels = np.zeros(n)
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]

                if split == 'training':
                    if not (input_labels != -1).any():
                        continue

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]

                # Finish one sample
                i += 1

                # In case batch is full, yield it and reset it
                if batch_n + n > batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),      # [BxN, 3]  - xyz in sample
                           np.concatenate(c_list, axis=0),      # [BxN, 3/1 + 3 (RGB/intensity + global xyz in whole cloud)]
                           np.concatenate(pl_list, axis=0),     # [BxN]     - labels
                           np.array([tp.shape[0] for tp in p_list]),    # [B]    - size (point num) of each batch
                           np.concatenate(pi_list, axis=0),             # [B, N] - point idx in each of its point cloud
                           np.array(ci_list, dtype=np.int32))           # [B]    - cloud idx
                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0
                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]
                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))
        spatially_regular_weaksup_gen.types = spatially_regular_gen.types
        spatially_regular_weaksup_gen.shapes = spatially_regular_gen.shapes

        # Define the generator that should be used for this split
        gen_func = config.data_gen if config.data_gen else 'spatially_regular_gen'
        gen_func = {
            'spatially_regular_gen': spatially_regular_gen,
            'spatially_regular_weaksup_gen': spatially_regular_weaksup_gen,
        }[gen_func]
        gen_types = tuple(gen_func.types)
        gen_shapes = tuple(gen_func.shapes)

        return gen_func, gen_types, gen_shapes

    def get_batch_gen_fixed_size(self, split):

        config = self.config
        epoch_n = self.init_sampling(split)
        # N = None
        N = config.in_points
        rgb_dims = self.rgb_dims if hasattr(self, 'rgb_dims') else 3

        def spatially_regular_gen():
            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=self.config.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                k = min(len(points), self.config.in_points)

                # Query all points / the predefined number within the cloud
                dists, input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=k)
                input_inds = input_inds[0]

                # Shuffle index
                np.random.shuffle(input_inds)

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[split][cloud_ind][input_inds]
                if split == 'test':
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]

                # Update potentials (Tuckey weights)
                # TODO: using dist from tree query ???
                # assert np.all(np.abs(dists ** 2 - np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)) < 1e-9)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / dists.max())
                # # weighted update
                # tukeys_cls_w = class_weight[split][input_labels] if split == 'train' else 1  # per-pt class weight
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # up_sampled with replacement
                if len(input_points) < self.config.in_points:
                    dup_idx = np.random.choice(len(points), self.config.in_points - len(points))
                    dup_idx = np.concatenate([np.arange(len(points)), dup_idx])  # [original, dup]
                    input_points = input_points[dup_idx]
                    input_colors = input_colors[dup_idx]
                    input_labels = input_labels[dup_idx]
                    input_inds = input_inds[dup_idx]

                # sampled point cloud
                yield (input_points.astype(np.float32),  # centered xyz
                        np.hstack([input_colors, input_points + pick_point]).astype(np.float32),  # colors, original xyz
                        input_labels,  # label
                        input_inds.astype(np.int32),  # points idx in cloud
                        int(cloud_ind)  # cloud idx
                        # np.array([cloud_ind], dtype=np.int32)
                    )
        spatially_regular_gen.types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        # after batch : [B, N, 3], [B, N, 6], [B, N], [B, N], [B]
        spatially_regular_gen.shapes = ([N, 3], [N, 3 + rgb_dims], [N], [N], [])

        def spatially_regular_weaksup_gen():
            # Generator loop
            i = 0
            while i < epoch_n:
                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))
                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])
                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_ind].data, copy=False)
                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                noise = np.random.normal(scale=self.config.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                # Check if the number of points in the selected cloud is less than the predefined num_points
                k = min(len(points), self.config.in_points)
                # Query all points / the predefined number within the cloud
                dists, input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=k)
                input_inds = input_inds[0]

                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / dists.max())
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                if split == 'training' and not any(i.startswith('pseudo') for i in config.architecture):
                    label_inds = np.where(~np.isin(self.input_labels[split][cloud_ind], self.ignored_labels))[0]  # all valid labels
                    # input_label_mask = ~np.isin(self.input_labels[split][cloud_ind][input_inds], self.ignored_labels)
                    # input_label_inds = input_inds[input_label_mask]  # current valid labels
                    # other_label_mask = ~np.isin(label_inds, input_label_inds)
                    # label_inds = label_inds[other_label_mask]  # concat others only
                    input_inds = np.concatenate([label_inds, input_inds])[:k]

                # Shuffle index
                np.random.shuffle(input_inds)
                # Get label
                if split == 'test':
                    input_labels = np.zeros(input_inds.shape[0])
                else:
                    input_labels = self.input_labels[split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]

                if split == 'training':
                    if (input_labels == -1).all():
                        continue
                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[split][cloud_ind][input_inds]

                i += 1

                # up_sampled with replacement
                if len(input_points) < self.config.in_points:
                    dup_idx = np.random.choice(len(points), self.config.in_points - len(points))
                    dup_idx = np.concatenate([np.arange(len(points)), dup_idx])  # [original, dup]
                    input_points = input_points[dup_idx]
                    input_colors = input_colors[dup_idx]
                    input_labels = input_labels[dup_idx]
                    input_inds = input_inds[dup_idx]
                # sampled point cloud
                yield (input_points.astype(np.float32),  # centered xyz
                        np.hstack([input_colors, input_points + pick_point]).astype(np.float32),  # colors, original xyz
                        input_labels,  # label
                        input_inds.astype(np.int32),  # points idx in cloud
                        int(cloud_ind)  # cloud idx
                    )
        spatially_regular_weaksup_gen.types = spatially_regular_gen.types
        spatially_regular_weaksup_gen.shapes = spatially_regular_gen.shapes

        # Define the generator that should be used for this split
        valid_split = ('training', 'validation', 'test')
        assert split in valid_split, ValueError(f'invalid split = {split} not in {valid_split}')
        gen_func = config.data_gen if config.data_gen else 'spatially_regular_gen'
        gen_func = {
            'spatially_regular_gen': spatially_regular_gen,
            'spatially_regular_weaksup_gen': spatially_regular_weaksup_gen,
        }[gen_func]
        gen_types = tuple(gen_func.types)
        gen_shapes = tuple(gen_func.shapes)

        return gen_func, gen_types, gen_shapes


    def tf_augment_input(self, stacked_points, batch_inds):
        """
        Augment inputs with rotation, scale and noise
        Args:
            batch_inds : [BxN] - batch idx for each point - from tf_get_batch_inds
        """
        # Parameter
        config = self.config
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########
        if config.augment_rotation == 'none' or not config.augment_rotation:
            R = tf.eye(3, batch_shape=(num_batches,))  # [B, 3, 3]
        elif config.augment_rotation == 'vertical':  # -- used in default cfgs
            # Choose a random angle for each element
            theta = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))  # [B, 3, 3]
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)  # [BxN, 3, 3]
            # Apply rotations
            if len(stacked_rots.shape) == len(stacked_points.shape):
                stacked_rots = tf.expand_dims(stacked_rots, axis=-3)  # [BxN, 1, 3, 3] to match [B, N, 1, 3]
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=-2),  # to row vec: [BxN, 3] -> [BxN, 1, 3]
                                                  stacked_rots),
                                        tf.shape(stacked_points))
        elif config.augment_rotation == 'arbitrarily':
            cs0 = tf.zeros((num_batches,))
            cs1 = tf.ones((num_batches,))
            # x rotation
            thetax = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cx, sx = tf.cos(thetax), tf.sin(thetax)
            Rx = tf.stack([cs1, cs0, cs0, cs0, cx, -sx, cs0, sx, cx], axis=1)
            Rx = tf.reshape(Rx, (-1, 3, 3))
            # y rotation
            thetay = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cy, sy = tf.cos(thetay), tf.sin(thetay)
            Ry = tf.stack([cy, cs0, -sy, cs0, cs1, cs0, sy, cs0, cy], axis=1)
            Ry = tf.reshape(Ry, (-1, 3, 3))
            # z rotation
            thetaz = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cz, sz = tf.cos(thetaz), tf.sin(thetaz)
            Rz = tf.stack([cz, -sz, cs0, sz, cz, cs0, cs0, cs0, cs1], axis=1)
            Rz = tf.reshape(Rz, (-1, 3, 3))
            # whole rotation
            Rxy = tf.matmul(Rx, Ry)
            R = tf.matmul(Rxy, Rz)
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)
            # Apply rotations
            if len(stacked_rots.shape) < len(stacked_points.shape):
                stacked_rots = tf.expand_dims(stacked_rots, axis=-3)  # [B, 1, 3, 3] to match [B, N, 1, 3]
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=-2), stacked_rots), tf.shape(stacked_points))
        else:
            raise ValueError('Unknown rotation augmentation : ' + config.augment_rotation)

        #######
        # Scale
        #######
        # Choose random scales for each example
        if config.augment_scale:
            min_s, max_s = 1 - config.augment_scale, 1 + config.augment_scale
        else:
            min_s, max_s = config.augment_scale_min, config.augment_scale_max
        if config.augment_scale_anisotropic:  # each batch a scale - [B, 3/1]
            s = tf.random.uniform((num_batches, 3), minval=min_s, maxval=max_s)  # xyz diff scale 
        else:
            s = tf.random.uniform((num_batches, 1), minval=min_s, maxval=max_s)  # xyz same scale
        if config.augment_symmetries:
            symmetries = []
            for i in range(3):
                if config.augment_symmetries is True or config.augment_symmetries[i]:  # could flip (multiply by 1/-1)
                    symmetries.append(tf.round(tf.random.uniform((num_batches, 1))) * 2 - 1)
                else:
                    symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
            s *= tf.concat(symmetries, 1)  # [B, 3]
        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)  # [BxN, 3]
        # Apply scales
        if len(stacked_scales.shape) < len(stacked_points.shape):
            stacked_scales = tf.expand_dims(stacked_scales, axis=-2)  # [B, 1, 3] to match [B, N, 3]
        stacked_points = stacked_points * stacked_scales

        #######
        # Noise
        #######
        noise = tf.random_normal(tf.shape(stacked_points), stddev=config.augment_noise)  # per-point noise
        stacked_points = stacked_points + noise
        return stacked_points, s, R

    def tf_augment_rgb_radius(self, stacked_colors, batch_inds, stacks_lengths):
        # stacked_colors - [BxN, 3]
        config = self.config
        num_batches = batch_inds[-1] + 1
        if config.augment_rgb_contrast:  # per-cloud rgb contrast - approx by using min/max across all selected cloud
            assert 0 < config.augment_rgb_contrast
            s = tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_contrast)
            blend_factor = [float(config.augment_rgb_contrast_bl)] * num_batches if config.augment_rgb_contrast_bl else tf.random.uniform([num_batches])
            b_starts = tf.cumsum(stacks_lengths, exclusive=True)
            b_ends = b_starts + stacks_lengths
            def contrast_rgb(bi, rgb):
                cur_rgb = stacked_colors[b_starts[bi]:b_ends[bi]]
                def true_fn():
                    cur_min = tf.reduce_min(cur_rgb, axis=0, keepdims=True)
                    cur_max = tf.reduce_max(cur_rgb, axis=0, keepdims=True)
                    cur_contrast = cur_rgb / (cur_max - cur_min)
                    bl = blend_factor[bi]
                    return (1 - bl) * cur_rgb + bl * cur_contrast
                c_rgb = tf.cond(s[bi], true_fn=true_fn, false_fn=lambda: cur_rgb)
                bi += 1
                rgb = tf.concat([rgb, c_rgb], axis=0)
                return bi, rgb
            _, stacked_colors = tf.while_loop(lambda bi, *args: bi < num_batches, contrast_rgb,
                loop_vars=[0, tf.zeros([0, 3], dtype=stacked_colors.dtype)], shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, 3])])
            # def true_fn():
            #     stacked_s = tf.gather(s, batch_inds)
            #     stacked_colors_select = tf.boolean_mask(stacked_colors, stacked_s)
            #     stacked_colors_min = tf.reduce_min(stacked_colors_select, axis=0, keepdims=True)
            #     stacked_colors_max = tf.reduce_max(stacked_colors_select, axis=0, keepdims=True)
            #     scale = stacked_colors_max - stacked_colors_min
            #     stacked_contrast = (stacked_colors - stacked_colors_min) / scale  # [BxN] - color in 0-1
            #     stacked_s = tf.expand_dims(tf.cast(stacked_s, tf.float32), axis=-1)
            #     return (1 - stacked_s) * stacked_colors + stacked_s * stacked_contrast  # no blending (but `s` 0/1 as on/off)
            # stacked_colors = tf.cond(tf.reduce_any(s), true_fn=true_fn, false_fn=lambda: stacked_colors)
        if config.augment_rgb_trans:  # per-cloud rgb translation
            assert 0 < config.augment_rgb_trans
            s = tf.cast(tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_trans), tf.float32)
            ratio = 0.05
            tr = (tf.random.uniform((num_batches, 3)) - 0.5) * 2 * ratio  # [-r, r]
            tr = tf.gather(tr * tf.expand_dims(s, axis=-1), batch_inds)
            stacked_colors = tf.clip_by_value(stacked_colors + tr, 0, 1)
        if config.augment_rgb_noise:  # per-point rgb noise
            assert 0 < config.augment_rgb_noise
            s = tf.cast(tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_noise), tf.float32)
            s = tf.expand_dims(tf.gather(s, batch_inds), axis=-1)
            stacked_noise = tf.random.normal(tf.shape(stacked_colors), stddev=0.005) * s
            stacked_colors = tf.clip_by_value(stacked_colors + stacked_noise, 0, 1)
        if config.augment_rgb_hstrans:  # per-point hue-saturation (in hsv space) trans
            assert 0 < config.augment_rgb_hstrans
            s = tf.cast(tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_hstrans), tf.float32)
            s = tf.expand_dims(s, axis=-1)
            ratio_h = tf.constant([0.5, 0, 0], dtype=tf.float32)  # + 0.5
            ratio_s = tf.constant([0, 0.2, 0], dtype=tf.float32)  # * [0.8, 1.2]
            stacked_hsv = tf.image.rgb_to_hsv(stacked_colors)
            ratio_h = tf.random.uniform((num_batches, 1)) * tf.expand_dims(ratio_h, axis=0)  # [B, 3]
            ratio_h = tf.gather(ratio_h * s, batch_inds)
            stacked_hsv = (stacked_hsv + ratio_h) // 1
            ratio_s = (tf.random.uniform((num_batches, 1)) - 0.5) * 2 * tf.expand_dims(ratio_s, axis=0)  # [B, 3]
            ratio_s = tf.gather(ratio_s * s, batch_inds)
            stacked_hsv = tf.clip_by_value(stacked_hsv * (ratio_s + 1), 0, 1)
            stacked_colors = tf.image.hsv_to_rgb(stacked_hsv)
        if config.augment_rgb_drop:  # per-cloud rgb drop
            s = tf.cast(tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_drop), tf.float32)
            stacked_s = tf.gather(s, batch_inds)
            stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)
        return stacked_colors


    def tf_augment_rgb_fixed_size(self, stacked_colors):
        # stacked_colors - [B, N, 3]
        config = self.config
        num_batches = tf.shape(stacked_colors)[0]
        if config.augment_rgb_contrast:  # per-cloud rgb contrast - approx by using min/max across all selected cloud
            assert 0 < config.augment_rgb_contrast
            s = tf.less(tf.random.uniform((num_batches,)), config.augment_rgb_contrast)
            blend_factor = float(config.augment_rgb_contrast_bl) if config.augment_rgb_contrast_bl else tf.random.uniform([num_batches, 1, 1])
            rgb_min = tf.reduce_min(stacked_colors, axis=1, keepdims=True)
            rgb_max = tf.reduce_max(stacked_colors, axis=1, keepdims=True)
            scale = rgb_max - rgb_min
            contrast_colors = (stacked_colors - rgb_min) / scale
            stacked_colors = (1 - blend_factor) * stacked_colors + blend_factor * contrast_colors
        if config.augment_rgb_trans:  # per-cloud rgb translation
            assert 0 < config.augment_rgb_trans
            s = tf.cast(tf.less(tf.random.uniform([num_batches, 1, 1]), config.augment_rgb_trans), tf.float32)
            ratio = 0.05
            tr = (tf.random.uniform([num_batches, 1, 3]) - 0.5) * 2 * ratio  # [-r, r]
            stacked_colors = tf.clip_by_value(stacked_colors + tr * s, 0, 1)
        if config.augment_rgb_noise:  # per-point rgb noise
            assert 0 < config.augment_rgb_noise
            s = tf.cast(tf.less(tf.random.uniform([num_batches, 1, 1]), config.augment_rgb_noise), tf.float32)
            stacked_noise = tf.random.normal(tf.shape(stacked_colors), stddev=0.005) * s
            stacked_colors = tf.clip_by_value(stacked_colors + stacked_noise, 0, 1)
        if config.augment_rgb_hstrans:  # per-point hue-saturation (in hsv space) trans
            assert 0 < config.augment_rgb_hstrans
            s = tf.cast(tf.less(tf.random.uniform([num_batches, 1, 1]), config.augment_rgb_hstrans), tf.float32)
            ratio_h = tf.constant([0.5, 0, 0], dtype=tf.float32)  # + 0.5
            ratio_s = tf.constant([0, 0.2, 0], dtype=tf.float32)  # * [0.8, 1.2]
            stacked_hsv = tf.image.rgb_to_hsv(stacked_colors)
            ratio_h = tf.random.uniform([num_batches, 1, 1]) * tf.reshape(ratio_h, [1, 1, 3])  # [B, 1, 3]
            stacked_hsv = (stacked_hsv + ratio_h * s) // 1
            ratio_s = (tf.random.uniform([num_batches, 1, 1]) - 0.5) * 2 * tf.reshape(ratio_s, [1, 1, 3])  # [B, 1, 3]
            stacked_hsv = tf.clip_by_value(stacked_hsv * (ratio_s + 1), 0, 1)
            stacked_colors = tf.image.hsv_to_rgb(stacked_hsv)
        if config.augment_rgb_drop:  # per-cloud rgb drop
            s = tf.cast(tf.less(tf.random.uniform([num_batches, 1, 1]), config.augment_rgb_drop), tf.float32)
            stacked_colors = stacked_colors * s
        return stacked_colors

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
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

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
                                                           tf.TensorShape([None])])

        return batch_inds

    def tf_stack_batch_inds(self, stacks_len, tight=True):
        """
        Stack the flat point idx, given the batch element sizes (stacks_len)
            E.g. stacks_len = [3, 2, 5]; n = sum(stacks_len) = 10
            => return: [[0, 1, 2, n, n, n], 
                        [3, 4, n, n, n, n],
                        [5, 6, 7, 8, 9, n]]
        """
        # Initiate batch inds tensor
        num_points = tf.reduce_sum(stacks_len)
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
        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=fixed_shapes)

        # Add a last column with shadow neighbor if there is not
        def f1(): return tf.pad(batch_inds, [[0, 0], [0, 1]], "CONSTANT", constant_values=num_points)
        def f2(): return batch_inds
        if not tight:
            batch_inds = tf.cond(tf.equal(num_points, max_points * tf.shape(stacks_len)[0]), true_fn=f1, false_fn=f2)

        return batch_inds

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """
        # crop neighbors matrix
        neighbors = neighbors[:, :self.neighborhood_limits[layer]]
        # neighbors = tf.reshape(neighbors, [-1, self.neighborhood_limits[layer]])
        return neighbors


    def tf_segmentation_inputs_radius(self,
                                      stacked_points,
                                      stacked_features,
                                      point_labels,
                                      stacks_lengths,
                                      batch_inds):
        if self.config.lazy_inputs:
            return self.tf_segmentation_inputs_lazy(stacked_points, stacked_features, point_labels, stacks_lengths, batch_inds)

        from ops import get_tf_func
        tf_batch_subsampling = get_tf_func(self.config.sample, verbose=self.verbose)
        tf_batch_neighbors = get_tf_func(self.config.search, verbose=self.verbose)

        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keepdims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)
        # Starting radius of convolutions
        dl = self.config.first_subsampling_dl
        dp = self.config.density_parameter
        r = dl * dp / 2.0
        # Lists of inputs
        num_layers = self.config.num_layers
        downsample_times = num_layers - 1
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)  # no upsample for input pt
        for dt in range(0, downsample_times):  # downsample times
            neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
            pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
            up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)

            neighbors_inds = self.big_neighborhood_filter(neighbors_inds, dt)
            pool_inds = self.big_neighborhood_filter(pool_inds, dt)
            up_inds = self.big_neighborhood_filter(up_inds, dt)

            input_points[dt] = stacked_points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            input_batches_len[dt] = stacks_lengths
            stacked_points = pool_points
            stacks_lengths = pool_stacks_lengths
            r *= 2
            dl *= 2

        # last (downsampled) layer points
        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, downsample_times)
        input_points[downsample_times] = stacked_points
        input_neighbors[downsample_times] = neighbors_inds
        input_pools[downsample_times] = tf.zeros((0, 1), dtype=tf.int32)
        input_batches_len[downsample_times] = stacks_lengths

        # Batch unstacking (with first layer indices for optional classif loss) - in_batches - input stage
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # Batch unstacking (with last layer indices for optional classif loss) - out_batches - most down-sampled stage
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        # list of network inputs
        input_dict = {
            'points': tuple(input_points),
            'neighbors': tuple(input_neighbors),
            'pools': tuple(input_pools),
            'upsamples': tuple(input_upsamples),
            'batches_len': tuple(input_batches_len),  # [[B], ...] - size of each point cloud in current batch
            'features': stacked_features,
            'batch_weights': stacked_weights,
            'in_batches': stacked_batch_inds_0,
            'out_batches': stacked_batch_inds_1,
            'in_batch_inds': batch_inds,
            'point_labels': point_labels,
        }

        return input_dict

    def tf_segmentation_inputs_fixed_size(self, points, features, point_labels):  # [B, N, 3], [B, N, d], [B, N]
        from ops import TF_OPS, get_tf_func

        config = self.config
        if config.lazy_inputs:
            return self.tf_segmentation_inputs_lazy(points, features, point_labels)

        assert config.sample in ['random', 'farthest'], f'not supported fixed-size sampling {self.config.sample}'
        assert config.search in ['knn'], f'not supported fixed-size neighbor searching {self.config.search}'
        sample_func = get_tf_func(config.sample, verbose=self.verbose)
        search_func = get_tf_func(config.search, verbose=self.verbose)

        num_layers = config.num_layers
        downsample_times = num_layers - 1

        # Lists of config
        k_search = config.kr_search if isinstance(config.kr_search, list) else [int(config.kr_search)] * num_layers  # k-nn for at each layer (stage)
        k_sample = config.kr_sample if isinstance(config.kr_sample, list) else [int(config.kr_sample)] * downsample_times  # k-nn for subsampling
        k_sample_up = config.kr_sample_up if isinstance(config.kr_sample_up, list) else [int(config.kr_sample_up)] * downsample_times  # k-nn for upsampling
        r_sample = config.r_sample if isinstance(config.r_sample, list) else [int(config.r_sample)] * downsample_times  # ratio for subsampling

        # Lists of inputs
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        n_points = self.config.in_points  # N at each layer (stage)
        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)  # no upsample for input pt
        for dt in range(0, downsample_times):
            neighbors_inds = search_func(points, points, k_search[dt])
            pool_points = sample_func(points, n_points // r_sample[dt])
            # pool_points = tf.gather(points, down_inds, batch_dims=1)
            pool_inds = search_func(pool_points, points, k_sample[dt])
            up_inds = search_func(points, pool_points, k_sample_up[dt])

            input_points[dt] = points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            points = pool_points
            n_points = int(pool_points.shape[-2]) if isinstance(pool_points.shape[-2].value, int) else tf.shape(pool_points)[-2]

        # last (downsampled) layer points
        dt = downsample_times
        neighbors_inds = search_func(points, points, k_search[dt])
        input_points[dt] = points
        input_neighbors[dt] = neighbors_inds
        input_pools[dt] = tf.zeros((0, 1), dtype=tf.int32)

        # # Batch unstacking (with first layer indices for optional classif loss) - in_batches
        # stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # # Batch unstacking (with last layer indices for optional classif loss) - out_batches
        # stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        # list of network inputs
        input_dict = {
            'points': tuple(input_points),
            'neighbors': tuple(input_neighbors),
            'pools': tuple(input_pools),
            'upsamples': tuple(input_upsamples),
            # 'batches_len': tuple(input_batches_len),
            'features': features,
            # 'batch_weights': stacked_weights,
            # 'in_batches': stacked_batch_inds_0,
            # 'out_batches': stacked_batch_inds_1,
            'point_labels': point_labels,
        }

        return input_dict

    def tf_segmentation_inputs_lazy(self, points, features, point_labels, stacks_lengths=None, batch_inds=None):
        config = self.config

        from ops import TF_OPS, get_tf_func
        sample_func = get_tf_func(config.sample, verbose=self.verbose)
        search_func = get_tf_func(config.search, verbose=self.verbose)

        num_layers = config.num_layers
        downsample_times = num_layers - 1

        # Lists of config
        kr_search = config.kr_search if isinstance(config.kr_search, list) else [int(config.kr_search)] * num_layers  # k-nn for at each layer (stage)
        # kr_sample = config.kr_sample if isinstance(config.kr_sample, list) else [int(config.kr_sample)] * downsample_times  # k-nn for subsampling
        # kr_sample_up = config.kr_sample_up if isinstance(config.kr_sample_up, list) else [int(config.kr_sample_up)] * downsample_times  # k-nn for upsampling
        # r_sample = config.r_sample if isinstance(config.r_sample, list) else [int(config.r_sample)] * downsample_times  # ratio for subsampling

        # Lists of inputs - filled with empty tensor
        emptyness = tf.zeros((0, 1), dtype=tf.int32)
        input_points = [emptyness] * num_layers
        input_neighbors = [emptyness] * num_layers
        input_pools = [emptyness] * num_layers
        input_upsamples = [emptyness] * num_layers

        # Prepare only the 1st stage
        input_points[0] = points
        if config.search in ['radius', 'knn']:
            input_neighbors[0] = TF_OPS.tf_fix_search(points, points, kr_search[0], config.search, stacks_lengths, stacks_lengths, verbose=False)

        # list of network inputs
        input_dict = {
            'points': tuple(input_points),
            'neighbors': tuple(input_neighbors),
            'pools': tuple(input_pools),
            'upsamples': tuple(input_upsamples),
            'features': features,
            'point_labels': point_labels,
        }

        if stacks_lengths is not None:
            # Batch weight at each point for loss (inverse of stacks_lengths for each point)
            min_len = tf.reduce_min(stacks_lengths, keepdims=True)
            batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
            stacked_weights = tf.gather(batch_weights, batch_inds)
            # Batch unstacking (with first layer indices for optional classif loss) - in_batches - input stage
            stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
            # batches_len
            input_batches_len = [stacks_lengths] + [emptyness] * [num_layers - 1]
            input_dict.update({
                'batches_len': tuple(input_batches_len),
                'batch_weights': stacked_weights,
                'in_batches': stacked_batch_inds_0,
                'out_batches': emptyness,
            })

        return input_dict

