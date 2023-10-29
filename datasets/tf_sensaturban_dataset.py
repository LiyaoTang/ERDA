
import os, re, sys, glob, time, pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.ply import read_ply, write_ply
from .base import Dataset

class SensatUrbanDataset(Dataset):

    label_to_names = {
        0: 'ground', 
        1: 'high vegetation', 
        2: 'buildings', 
        3: 'walls',
        4: 'bridge', 
        5: 'parking', 
        6: 'rail', 
        7: 'traffic roads', 
        8: 'street furniture',
        9: 'cars', 
        10: 'footpath', 
        11: 'bikes', 
        12: 'water',
        # 13 : empty,
    }
    ignored_labels = np.array([])  # 13

    label_to_rgb = np.array([  # actually, idx to rgb (invalid mapped to -1)
        [85, 107, 47],      # ground            ->  OliveDrab
        [0, 255, 0],        # tree              ->  Green
        [255, 165, 0],      # building          ->  orange
        [41, 49, 101],      # Walls             ->  darkblue
        [0, 0, 0],          # Bridge            ->  black
        [0, 0, 255],        # parking           ->  blue
        [255, 0, 255],      # rail              ->  Magenta
        [200, 200, 200],    # traffic Roads     ->  grey
        [89, 47, 95],       # Street Furniture  ->  DimGray
        [255, 0, 0],        # cars              ->  red
        [255, 255, 0],      # Footpath          ->  deeppink
        [0, 255, 255],      # bikes             ->  cyan
        [0, 191, 255],      # water             ->  skyblue
        # [0, 0, 0,],         # empty             ->  dark
    ])

    def __init__(self, config, verbose=True, input_threads=8, parse=True):
        super(SensatUrbanDataset, self).__init__(config)
        self.config = config
        self.verbose = verbose

        if config.padding or config.weak_supervise:
            self.label_to_names[13] = 'empty'
            self.ignored_labels = np.concatenate([self.ignored_labels, [13]], axis=0)
            self.label_to_rgb = np.concatenate([self.label_to_rgb, [[0, 0, 0]]], axis=0)

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # Number of input threads
        self.num_threads = max([input_threads, config.gpu_num, config.num_threads])

        # files
        self.all_files = np.sort(glob.glob(os.path.join(self.data_path, 'ply/*/*.ply')))  # ply / [train, test] / *.ply
        self.test_files = np.sort(glob.glob(os.path.join(self.data_path, 'ply/test/*.ply')))  # ply / test / *.ply
        self.val_file_name = ['birmingham_block_1',
                              'birmingham_block_5',
                              'cambridge_block_10',
                              'cambridge_block_7']
        self.test_file_name = ['birmingham_block_2', 'birmingham_block_8',
                               'cambridge_block_15', 'cambridge_block_22',
                               'cambridge_block_16', 'cambridge_block_27']
        # 1 to do validation, 2 to train on all data, 3 to train-val on all data
        self.validation_split = int(config.validation_split) if config.validation_split else 1
        assert self.validation_split in [0, 1, 2, 3]
        if self.validation_split == 0:  # 3 broken ply - contain much less pts than what specified in their header => seems no need to remove?
            self.all_files = [f for f in self.all_files if not any(f.endswith(n) for n in ['cambridge_block_0.ply', 'cambridge_block_1.ply', 'cambridge_block_34.ply'])]

        # Load test?
        self.load_test = 'test' in config.mode

        # prepare ply file
        self.prepare_ply(verbose=verbose)

        # input subsampling
        self.load_subsampled_clouds(config.first_subsampling_dl, verbose=verbose)

        # # initialize op - manual call
        # self.initialize(verbose=verbose)
        return

    def prepare_ply(self, verbose=True):
        # NOTE: ply already in [x, y, z, red, green, blue, /class] format
        pass

    def load_subsampled_clouds(self, subsampling_parameter=0.2, verbose=True):
        config = self.config

        sample_type = config.first_subsampling_type if config.first_subsampling_type else 'grid'
        if sample_type == 'grid':  # grid subsampling - default
            input_folder = f'input_{subsampling_parameter:.3f}'
            from ops import get_tf_func
            subsampling_func = get_tf_func('grid_preprocess')

        elif sample_type == 'rand':  # random subsampling
            input_folder = f'input_rand_{subsampling_parameter}'
            # args = [points, features=None, labels=None, subsample_ratio]
            subsampling_func = lambda *args: [x[np.random.choice(len(x), size=int(len(x) / args[-1]), replace=False)] for x in args[:-1] if x is not None]

        else:
            raise ValueError(f'not support sample_type={sample_type} - {self.__class__}')

        input_folder = os.path.join(self.data_path, input_folder)
        os.makedirs(input_folder, exist_ok=True)

        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}

        if verbose:
            print(f'\nPreparing KDTree for all scenes, subsampled into {input_folder}')
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split(os.sep)[-1][:-4]
            if cloud_name in self.test_file_name:
                cloud_split = 'test'
            elif cloud_name in self.val_file_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            if cloud_split == 'test' and not self.load_test:
                continue

            # Name of the input files
            kd_tree_file = os.path.join(input_folder, f'{cloud_name}_KDTree.pkl')
            sub_ply_file = os.path.join(input_folder, f'{cloud_name}.ply')

            if verbose:
                action = 'Found' if os.path.isfile(kd_tree_file) else 'Preparing'
                print(f'{action} KDTree for cloud {cloud_name} - ', end='')

            # subsample ply
            if not os.path.exists(sub_ply_file):
                # do subsampling
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class'] if cloud_split != 'test' else None
                # calling with - (points, features=None, labels=None, sampleDl=0.1) 
                sub_data_list = subsampling_func(points, colors, labels, subsampling_parameter)
                sub_points, sub_colors = sub_data_list[:2]
                # save
                if cloud_split != 'test':
                    sub_labels = sub_data_list[2]
                    write_ply(sub_ply_file, [sub_points, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
                else:
                    sub_labels = None
                    write_ply(sub_ply_file, [sub_points, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
            else:
                # load
                sub_data = read_ply(sub_ply_file)
                sub_points = None
                sub_colors = np.vstack((sub_data['red'], sub_data['green'], sub_data['blue'])).T
                sub_labels = sub_data['class'] if cloud_split != 'test' else None
            sub_colors /= 255.0

            # kd-tree
            if not os.path.exists(kd_tree_file):
                # build kd-tree
                if sub_points is None:
                    sub_points = np.vstack((sub_data['x'], sub_data['y'], sub_data['z'])).T
                search_tree = KDTree(sub_points, leaf_size=50)
                # save
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)
            else:
                # load
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            if verbose:
                size = sub_colors.shape[0] * 4 * 7
                print(f'{size * 1e-6:.1f} MB loaded in {time.time() - t0:.1f}s')

        if self.validation_split in [2, 3]:  # train on all
            self.input_trees['training'] += self.input_trees['validation']
            self.input_colors['training'] += self.input_colors['validation']
            self.input_labels['training'] += self.input_labels['validation']
            self.input_names['training'] += self.input_names['validation']

        if self.validation_split == 3:  # train-val on all
            self.input_trees['validation'] = self.input_trees['training']
            self.input_colors['validation'] = self.input_colors['training']
            self.input_labels['validation'] = self.input_labels['training']
            self.input_names['validation'] = self.input_names['training']

        # Sub-sample to weak supervision
        if self.config.weak_supervise:
            self.input_labels['training'] = [self.get_weak_supervision(sub_labels) for sub_labels in self.input_labels['training']]

        def get_proj_idx(search_tree, file_path, proj_file):
            if not os.path.exists(proj_file):
                data = read_ply(file_path)
                xyz = np.vstack((data['x'], data['y'], data['z'])).T
                labels = data['class'] if 'train' in file_path else np.zeros(xyz.shape[0], dtype=np.int32)
                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)
            else:
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
            return proj_idx, labels

        # Get reprojected indices
        # - val projection and labels
        if verbose:
            print('\nPreparing reprojected indices for validation - ', end='')
            t0 = time.time()
        self.validation_proj = []
        self.validation_labels = []
        for cloud_name, search_tree in zip(self.input_names['validation'], self.input_trees['validation']):
            proj_file = os.path.join(input_folder, f'{cloud_name}_proj.pkl')
            file_path = os.path.join(self.data_path, f'ply/train/{cloud_name}.ply')

            proj_idx, labels = get_proj_idx(search_tree, file_path, proj_file)
            self.validation_proj += [proj_idx]
            self.validation_labels += [labels]
        if verbose:
            print(f'done in {time.time() - t0:.1f}s\n')

        # - val statstics
        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])

        # - test projection and labels
        if not self.load_test:
            return
        if verbose:
            print('Preparing reprojected indices for test - ', end='')
            t0 = time.time()
        self.test_proj = []
        self.test_labels = []
        for cloud_name, search_tree in zip(self.input_names['test'], self.input_trees['test']):
            proj_file = os.path.join(input_folder, f'{cloud_name}_proj.pkl')
            file_path = os.path.join(self.data_path, f'ply/test/{cloud_name}.ply')

            proj_idx, labels = get_proj_idx(search_tree, file_path, proj_file)
            self.test_proj += [proj_idx]
            self.test_labels += [labels]
        if verbose:
            print(f'done in {time.time() - t0:.1f}s\n')
        return


    def get_tf_mapping_radius(self):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            """
            Args:
                stacked_points  : [BxN, 3]  - xyz in sample
                stacked_colors  : [BxN, 6]  - rgb,  xyz in whole cloud (global xyz)
                point_labels    : [BxN]     - labels
                stacks_lengths  : [B]    - size (point num) of each batch
                point_inds      : [B, N] - point idx in each of its point cloud
                cloud_inds      : [B]    - cloud idx
            """
            config = self.config

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)  # [BxN]

            # Augmentation: input points (xyz) - rotate, scale (flip), jitter
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]
            stacked_rgb = stacked_colors

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features:
                stacked_colors = self.tf_augment_rgb_radius(stacked_colors, batch_inds, stacks_lengths=stacks_lengths)

            stacked_fin = []  # default to '1rgbZ'
            if not (in_features and re.fullmatch('(|1)(|rgb)(|xyz)(|XYZ)(|Z)', in_features)):
                raise ValueError(f'not support in_features = {in_features}')
            if '1' in in_features:
                stacked_fin += [tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)]
            if 'rgb' in in_features:
                stacked_fin += [stacked_colors]
            if 'xyz' in in_features:
                stacked_fin += [stacked_points]
            if 'XYZ' in in_features:
                stacked_fin += [stacked_original_coordinates]
            elif 'Z' in in_features:
                stacked_fin += [stacked_original_coordinates[..., 2:]]
            stacked_features = tf.concat(stacked_fin, axis=-1) if len(stacked_fin) > 1 else stacked_fin[0]

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_radius(stacked_points,
                                                        stacked_features,
                                                        point_labels,
                                                        stacks_lengths,
                                                        batch_inds)
            if isinstance(inputs, dict):
                inputs.update({
                    'rgb': stacked_rgb,
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
            else:
                # Add scale and rotation for testing
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]
            return inputs

        return tf_map

    def get_tf_mapping_fixed_size(self):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, point_inds, cloud_inds):
            """
            Args:
                stacked_points  : [B, N, 3]  - xyz in sample
                stacked_colors  : [B, N, 6]  - rgb,  xyz in whole cloud (global xyz)
                point_labels    : [B, N]     - labels
                point_inds      : [B, N] - point idx in each of its point cloud
                cloud_inds      : [B]    - cloud idx
            """
            config = self.config

            # Get batch indice for each point
            BN = tf.shape(stacked_points)
            batch_inds = tf.range(BN[0])  # [B] - may have leftover sample (to avoid: set drop_reminder=True)

            # Augmentation: input points (xyz) - rotate, scale (flip), jitter
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((BN[0], BN[1], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, :, 3:]
            stacked_colors = stacked_colors[:, :, :3]
            stacked_rgb = stacked_colors

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features:
                stacked_colors = self.tf_augment_rgb_fixed_size(stacked_colors)

            stacked_fin = []  # default to '1rgbZ'
            if not (in_features and re.fullmatch('(|1)(|rgb)(|xyz)(|XYZ)(|Z)', in_features)):
                raise ValueError(f'not support in_features = {in_features}')
            if '1' in in_features:
                # First add a column of 1 as feature for the network to be able to learn 3D shapes
                stacked_fin += [tf.ones((BN[0], BN[1], 1), dtype=tf.float32)]
            if 'rgb' in in_features:
                stacked_fin += [stacked_colors]
            if 'xyz' in in_features:
                stacked_fin += [stacked_points]
            if 'XYZ' in in_features:
                stacked_fin += [stacked_original_coordinates]
            elif 'Z' in in_features:
                stacked_fin += [stacked_original_coordinates[..., 2:]]
            stacked_features = tf.concat(stacked_fin, axis=-1) if len(stacked_fin) > 1 else stacked_fin[0]

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_fixed_size(stacked_points,    # [B, N, 3]
                                                            stacked_features,  # [B, N, d]
                                                            point_labels)
            # Add scale and rotation for testing
            if isinstance(inputs, dict):
                inputs.update({
                    'rgb': stacked_rgb,
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
            else:
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]
            return inputs

        return tf_map

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """
        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T

    def get_weak_supervision(self, sub_labels):
        sub_labels = sub_labels.copy()
        # sample for each class
        for k in self.label_values:
            pts_inds = np.where(sub_labels == k)[0]  # inds of class-k pts in point cloud
            num_pts = len(pts_inds)
            if num_pts == 0:
                continue
            if 'pt' in self.config.weak_supervise:
                num_label = int(self.config.weak_supervise[:-2])
            else:
                r = float(self.config.weak_supervise.replace('%', ''))
                r = r if '%' not in self.config.weak_supervise else r / 100
                if not (0 < r and r < 1):
                    raise ValueError(f'invalid r={r} from {self.config.weak_supervise}')
                num_label = max(int(num_pts * r), 1)
            inds = np.arange(num_pts)  # inds into pts_inds
            np.random.shuffle(inds)
            label_inds = pts_inds[inds[:num_label]]  # inds into sub_labels - remaining
            unsup_inds = pts_inds[inds[num_label:]]
            sub_labels[unsup_inds] = self.ignored_labels[0]
        return sub_labels


if __name__ == '__main__':

    from config.sensaturban import default as config

    dataset = SensatUrbanDataset(config)
    dataset.initialize()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(dataset.train_init_op)
        while True:
            inputs = sess.run(dataset.flat_inputs[0])
            xyz = inputs['points'][0]
            sub_xyz = inputs['points'][1]
            label = inputs['point_labels']
            raise
