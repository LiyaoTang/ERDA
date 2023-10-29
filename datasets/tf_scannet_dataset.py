# Basic libs
import os, re, sys, time, json, pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh
from .base import Dataset


class ScanNetDataset(Dataset):
    """
    Class to handle ScanNet dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    # Dict from labels to names
    label_to_names = {0: 'unclassified',
                      1: 'wall',
                      2: 'floor',
                      3: 'cabinet',
                      4: 'bed',
                      5: 'chair',
                      6: 'sofa',
                      7: 'table',
                      8: 'door',
                      9: 'window',
                      10: 'bookshelf',
                      11: 'picture',
                      12: 'counter',
                      14: 'desk',
                      16: 'curtain',
                      24: 'refridgerator',
                      28: 'shower curtain',
                      33: 'toilet',
                      34: 'sink',
                      36: 'bathtub',
                      39: 'otherfurniture'}
    ignored_labels = np.sort([0])

    def __init__(self, config, verbose=True, input_threads=8):
        super(ScanNetDataset, self).__init__(config)
        self.verbose = verbose

        self.version = str(config.version)
        if self.version:
            raise ValueError(f'not support version={self.version}')

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # Number of input threads
        self.num_threads = max([input_threads, config.gpu_num, config.num_threads])

        # Path of the folder containing ply files
        self.path = self.data_path
        # Path of the training files
        self.train_path = os.path.join(self.path, f'training_points{self.version}')
        self.test_path = os.path.join(self.path, f'test_points{self.version}')

        # Prepare ply files
        self.prepare_pointcloud_ply(verbose=verbose)

        # List of training and test files
        self.train_files = np.sort([os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f[-4:] == '.ply'])
        self.test_files = np.sort([os.path.join(self.test_path, f) for f in os.listdir(self.test_path) if f[-4:] == '.ply'])
        # Proportion of validation scenes
        self.validation_clouds = np.loadtxt(os.path.join(self.path, 'scannetv2_val.txt'), dtype=np.str)
        self.all_splits = []

        # 1 to do validation, 2 to train on all data, 3 to train-val on all data
        self.validation_split = config.validation_split if config.validation_split else 1
        assert self.validation_split in [1, 2, 3]
        # Load test?
        self.load_test = 'test' in config.mode

        # input subsampling
        self.load_subsampled_clouds(config.first_subsampling_dl, verbose=verbose)

        # # initialize op - manual call
        # self.initialize(verbose=verbose)
        return

    def prepare_pointcloud_ply(self, verbose):

        if self.config.prepare_mp:
            self.prepare_pointcloud_ply_mp(verbose)  # no return: check after multi-processing

        # Folder for the ply files
        paths = [os.path.join(self.path, 'scans'), os.path.join(self.path, 'scans_test')]
        new_paths = [self.train_path, self.test_path]
        mesh_paths = [os.path.join(self.path, f'training_meshes{self.version}'), os.path.join(self.path, f'test_meshes{self.version}')]

        # Mapping from annot to labels ID
        label_files = os.path.join(self.path, 'scannetv2-labels.combined.tsv')
        id_col = {
            '_200': 0,  # new id
            '': 4,  # nyuID
        }[self.version]
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[id_col]) for line in lines[1:]]
            annot_to_ID = {n: id for n, id in zip(names1, IDs)}

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        if verbose:
            print(f'\nPreparing ply files into {new_paths}')
            t0 = time.time()

        for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):

            # Create folder
            os.makedirs(new_path, exist_ok=True)
            os.makedirs(mesh_path, exist_ok=True)

            # Get scene names
            scenes = np.sort([f for f in os.listdir(path)])
            N = len(scenes)

            for i, scene in enumerate(scenes):

                #############
                # Load meshes
                #############

                # Check if file already done
                if os.path.exists(os.path.join(new_path, scene + '.ply')):
                    continue
                t1 = time.time()

                # Read mesh
                vertex_data, faces = read_ply(os.path.join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
                vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

                vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
                if new_path == self.train_path:

                    # Load alignment matrix to realign points
                    align_mat = None
                    with open(os.path.join(path, scene, scene + '.txt'), 'r') as txtfile:
                        lines = txtfile.readlines()
                    for line in lines:
                        line = line.split()
                        if line[0] == 'axisAlignment':
                            align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
                    R = align_mat[:3, :3]
                    T = align_mat[:3, 3]
                    vertices = vertices.dot(R.T) + T

                    # Get objects segmentations - to generate segmentation annotations
                    with open(os.path.join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                        segmentations = json.load(f)

                    segIndices = np.array(segmentations['segIndices'])  # per-vertex index of mesh segment

                    # Get objects classes
                    with open(os.path.join(path, scene, scene + '.aggregation.json'), 'r') as f:
                        aggregation = json.load(f)

                    # Loop on object to classify points
                    for segGroup in aggregation['segGroups']:
                        c_name = segGroup['label']
                        if c_name in names1:
                            nyuID = annot_to_ID[c_name]
                            if nyuID in self.label_values:
                                for segment in segGroup['segments']:  # vertex index of current object
                                    vertices_labels[segIndices == segment] = nyuID

                    # TODO: directly use the '_vh_clean_2.labels.ply' ?

                    # Save mesh
                    write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors, vertices_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                              triangular_faces=faces)

                else:
                    # Save mesh
                    write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'],
                              triangular_faces=faces)

                ###########################
                # Create finer point clouds
                ###########################

                # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
                points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

                # Subsample points
                sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

                # Collect colors from associated vertex
                sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

                if new_path == self.train_path:

                    # Collect labels from associated vertex
                    sub_labels = vertices_labels[sub_vert_inds.ravel()]

                    # Save points
                    write_ply(os.path.join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

                else:

                    # Save points
                    write_ply(os.path.join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

                if verbose:
                    print('{:s}/{:s} {:.1f} sec  / {:.1f}%'.format(new_path, scene, time.time() - t1, 100 * i / N))
        if verbose:
            print('Done in {:.1f}s'.format(time.time() - t0))

    def prepare_pointcloud_ply_mp(self, verbose):
        import multiprocessing as mp
        prepare_mp = mp.cpu_count() if self.config.prepare_mp == 'a' else int(self.config.prepare_mp)
        prepare_mp = prepare_mp if prepare_mp > 1 else self.num_threads

        # Folder for the ply files
        paths = [os.path.join(self.path, 'scans'), os.path.join(self.path, 'scans_test')]
        new_paths = [self.train_path, self.test_path]
        mesh_paths = [os.path.join(self.path, f'training_meshes{self.version}'), os.path.join(self.path, f'test_meshes{self.version}')]

        # Mapping from annot to labels ID
        label_files = os.path.join(self.path, 'scannetv2-labels.combined.tsv')
        id_col = {
            '_200': 0,  # new id
            '': 4,  # nyuID
        }[self.version]
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[id_col]) for line in lines[1:]]
            annot_to_ID = {n: id for n, id in zip(names1, IDs)}

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        if verbose:
            print(f'\nPreparing ply files into {new_paths} - with multi-processing {prepare_mp}')
            t0 = time.time()
        for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):
            # Create folder
            os.makedirs(new_path, exist_ok=True)
            os.makedirs(mesh_path, exist_ok=True)
            # Get scene names
            scenes = np.sort([f for f in os.listdir(path)])
            N = len(scenes)

            global _ScanNetDataset__prepare_scene  # mimic a private member
            def _ScanNetDataset__prepare_scene(i, scene):
                # Check if file already done
                if os.path.exists(os.path.join(new_path, scene + '.ply')):
                    return
                t1 = time.time()
                # -- Load mesh --
                # Read mesh
                vertex_data, faces = read_ply(os.path.join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
                vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T
                vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
                if new_path == self.train_path:
                    # Load alignment matrix to realign points
                    align_mat = None
                    with open(os.path.join(path, scene, scene + '.txt'), 'r') as txtfile:
                        lines = txtfile.readlines()
                    for line in lines:
                        line = line.split()
                        if line[0] == 'axisAlignment':
                            align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
                    R = align_mat[:3, :3]
                    T = align_mat[:3, 3]
                    vertices = vertices.dot(R.T) + T
                    # Get objects segmentations - to generate segmentation annotations
                    with open(os.path.join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                        segmentations = json.load(f)
                    segIndices = np.array(segmentations['segIndices'])  # per-vertex index of mesh segment
                    # Get objects classes
                    with open(os.path.join(path, scene, scene + '.aggregation.json'), 'r') as f:
                        aggregation = json.load(f)
                    # Loop on object to classify points
                    for segGroup in aggregation['segGroups']:
                        c_name = segGroup['label']
                        if c_name in names1:
                            nyuID = annot_to_ID[c_name]
                            if nyuID in self.label_values:
                                for segment in segGroup['segments']:  # vertex index of current object
                                    vertices_labels[segIndices == segment] = nyuID
                    # TODO: directly use the '_vh_clean_2.labels.ply' ?
                    # Save mesh
                    write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors, vertices_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                              triangular_faces=faces)
                else:
                    # Save mesh
                    write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'],
                              triangular_faces=faces)
                # -- Create finer point clouds --
                # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
                points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)
                # Subsample points
                sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)
                # Collect colors from associated vertex
                sub_colors = vertices_colors[sub_vert_inds.ravel(), :]
                if new_path == self.train_path:
                    # Collect labels from associated vertex
                    sub_labels = vertices_labels[sub_vert_inds.ravel()]
                    # Save points
                    write_ply(os.path.join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])
                else:
                    # Save points
                    write_ply(os.path.join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                if verbose:
                    print('{:s}/{:s} {:.1f} sec  / {:.1f}%'.format(new_path, scene, time.time() - t1, 100 * i / N))

            with mp.Pool(prepare_mp) as p:
                p.starmap(_ScanNetDataset__prepare_scene, list(zip(range(N), scenes)))
            del globals()['_ScanNetDataset__prepare_scene']  # delete the fake private member

        if verbose:
            print('Done in {:.1f}s'.format(time.time() - t0))
        return

    def load_subsampled_clouds(self, subsampling_parameter, verbose=True):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        # Create path for files
        tree_path = os.path.join(self.path, f'input_{subsampling_parameter:.3f}{self.version}')
        os.makedirs(tree_path, exist_ok=True)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initiate containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_vert_inds = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}

        # Advanced display
        N = len(self.train_files) if not self.load_test else len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        if verbose:
            print(f'\nPreparing KDTree for all scenes, into {tree_path}')
        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # get cloud name (scene id) and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if cloud_name in self.validation_clouds:
                    self.all_splits += [1]
                    cloud_split = 'validation'
                else:
                    self.all_splits += [0]
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            # if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
            if cloud_split == 'test' and not self.load_test:
                continue

            # Name of the input files
            KDTree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if os.path.isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack((data['vert_ind'], data['class'])).T

                # Subsample cloud
                sub_points, sub_colors, sub_int_features = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=int_features,
                                                                      sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if self.config.debug_half_color:  # rescale to [-1, 1]
                    sub_colors = sub_colors * 2 - 1
                if cloud_split == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features)
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if cloud_split == 'test':
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_vert_inds[cloud_split] += [sub_vert_inds]
            self.input_names[cloud_split] += [(cloud_name, cloud_folder)]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            if verbose and sys.stdout.isatty():
                print('', end='\r')
                print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

        if self.validation_split in [2, 3]:  # add val into train
            self.input_trees['training'] += self.input_trees['validation']
            self.input_colors['training'] += self.input_colors['validation']
            self.input_vert_inds['training'] += self.input_vert_inds['validation']
            self.input_names['training'] += self.input_names['validation']
            self.input_labels['training'] += self.input_labels['validation']

        if self.validation_split == 3:  # use all train-val as val
            self.input_trees['validation'] = self.input_trees['training']
            self.input_colors['validation'] = self.input_colors['training']
            self.input_vert_inds['validation'] = self.input_vert_inds['training']
            self.input_names['validation'] = self.input_names['training']
            self.input_labels['validation'] = self.input_labels['training']
            self.validation_clouds = [c for c, _ in self.input_names['validation']]

        # convert to weak supervision
        if self.config.weak_supervise:
            self.input_labels['training'] = self.get_weak_supervision_by_split('training', tree_path)

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = [None] * self.num_validation
        self.validation_labels = [None] * self.num_validation
        self.test_proj = [None] * self.num_test
        self.test_labels = [None] * self.num_test
        i_finished = 0

        # Advanced display
        N = self.num_validation + self.num_test
        if verbose and sys.stdout.isatty():
            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), flush=True)

        if verbose:
            print('\nPreparing reprojection indices for validation and test')
        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if (cloud_name, cloud_folder) in self.input_names['validation']:
                i_val = self.input_names['validation'].index((cloud_name, cloud_folder))
                proj_file = os.path.join(tree_path, f'{cloud_name}_proj.pkl')
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = mesh_path[-2].replace('points', 'meshes')
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = vertex_data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj[i_val] = proj_inds
                self.validation_labels[i_val] = labels
                i_finished += 1

            # Test projection
            if self.load_test and (cloud_name, cloud_folder) in self.input_names['test']:
                i_test = self.input_names['test'].index((cloud_name, cloud_folder))
                proj_file = os.path.join(tree_path, f'{cloud_name}_proj.pkl')
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = mesh_path[-2].replace('points', 'meshes')
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = np.zeros(vertices.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj[i_test] = proj_inds
                self.test_labels[i_test] = labels
                i_finished += 1

            if verbose and sys.stdout.isatty():
                print('', end='\r')
                print(fmt_str.format('#' * ((i_finished * progress_n) // N), 100 * i_finished / N), end='', flush=True)

        # class cnt over all clouds in val set (w/o invalid class)
        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])

        if verbose:
            print('\n')

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

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
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]
            stacked_rgb = stacked_colors

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features:
                stacked_colors = self.tf_augment_rgb_radius(stacked_colors, batch_inds, stacks_lengths=stacks_lengths)

            # Then use positions or not
            if in_features == '1':
                pass
            elif in_features == '1Z':
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[..., 2:]), axis=1)
            elif in_features == 'rgb':
                stacked_features = stacked_colors
            elif in_features == '1rgb':
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif in_features == '1rgbZ':  # used in provided cfgs - [1, rgb, z]
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[..., 2:]), axis=1)
            elif in_features == '1rgbxyz':
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            elif in_features == '1rgbxyzZ':
                stacked_features = tf.concat([stacked_features, stacked_colors, stacked_points, stacked_original_coordinates[..., 2:]], axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

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
        config = self.config
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
                raise ValueError(f'Not accepting in features = {in_features}')
            if '1' in in_features:
                # First add a column of 1 as feature for the network to be able to learn 3D shapes
                stacked_fin += [tf.ones((BN[0], BN[1], 1), dtype=tf.float32)]
            if 'rgb' in in_features:
                stacked_fin += [stacked_colors]
            if 'xyz' in in_features:
                stacked_fin += [stacked_points]
            if 'XYZ' in in_features:
                stacked_fin += [stacked_original_coordinates]
            if 'Z' in in_features and 'XYZ' not in in_features:
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

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Evaluation points are from coarse meshes, not from the ply file we created for our own training
        mesh_path = file_path.split('/')
        mesh_path[-2] = mesh_path[-2][:-6] + 'meshes'
        mesh_path = '/'.join(mesh_path)
        vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
        return np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T

    def get_weak_supervision_by_split(self, split, tree_path):
        weak_labels = []
        for (cloud_n, cloud_f), search_tree, sub_labels in zip(self.input_names[split], self.input_trees[split], self.input_labels[split]):
            sub_labels = self.get_weak_supervision(cloud_n, cloud_f, tree_path, search_tree, sub_labels)
            weak_labels.append(sub_labels)
        return weak_labels


    def get_weak_supervision(self, cloud_name, cloud_folder, tree_path, search_tree, sub_labels):
        weak_type = self.config.weak_supervise
        weak_prop = 'k'
        if '-' in weak_type:
            weak_type, weak_prop = weak_type.split('-')
        sub_labels = sub_labels.copy()

        # self-generated sparse label
        if 'pt' in weak_type or '%' in weak_type or float(weak_type) < 1:
            for k in self.label_values:
                if k in self.ignored_labels:
                    continue
                pts_inds = np.where(sub_labels == k)[0]  # inds of class-k pts in point cloud
                num_pts = len(pts_inds)
                if num_pts == 0:
                    continue
                if 'pt' in weak_type:
                    num_label = int(weak_type[:-2])
                else:
                    r = float(weak_type.replace('%', ''))
                    r = r if '%' not in weak_type else r / 100
                    if not (0 < r and r < 1):
                        raise ValueError(f'invalid r={r} from {weak_type}')
                    num_label = max(int(num_pts * r), 1)
                inds = np.arange(num_pts)  # inds into pts_inds
                np.random.shuffle(inds)
                label_inds = pts_inds[inds[:num_label]]  # inds into sub_labels - remaining
                unsup_inds = pts_inds[inds[num_label:]]
                sub_labels[unsup_inds] = 0
            return sub_labels

        # scannet data efficient
        if not hasattr(self, 'weaksup_pts'):
            # load preprocessed npz dict - from 'preprocess_weaksup'
            self.weaksup_pts = np.load(os.path.join(self.data_path, f'tasks/points/points{weak_type}.npz'))
        label_inds = self.weaksup_pts[cloud_name] if cloud_name in self.weaksup_pts else []  # vertices inds with annotation

        mesh_path = os.path.join(self.data_path, cloud_folder.replace('points', 'meshes'), cloud_name + '_mesh.ply')
        vertex_data, faces = read_ply(mesh_path, triangular_mesh=True)
        vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
        weak_pts = vertices[label_inds]
        weak_labels = vertex_data['class'][label_inds]

        if weak_pts.size == 0:
            weak_inds = []
            assert (cloud_name, cloud_folder) in self.input_names['validation']

        elif weak_prop.startswith('r'):
            # prop to a small radius
            r = float(weak_prop[1:]) if weak_prop[1:] else self.config.first_subsampling_dl / 2
            weak_inds = [np.squeeze(i).astype(np.int32) for i in search_tree.query_radius(weak_pts, r=r)]
            assert all(len(i) > 0 for i in weak_inds)  # at least one pts

        elif weak_prop.startswith('k'):
            # neares-k
            k = int(weak_prop[1:]) if weak_prop and weak_prop[1:] else 1
            weak_inds = np.squeeze(search_tree.query(weak_pts, k=k, return_distance=False))
            # proj_inds, full_labels = self.get_projection(cloud_name, cloud_folder, tree_path, search_tree=search_tree)  # proj_inds - [#full pts]
            # weak_inds = proj_inds[label_inds]
            # weak_labels = full_labels[label_inds]
        else:
            raise ValueError(f'not support weak_prop={weak_prop}')

        sub_labels[:] = 0
        for i, sub_inds in enumerate(weak_inds):
            sub_labels[sub_inds] = weak_labels[i]
        return sub_labels

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        n_b = config.batch_num
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d} => b = {:.1f}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1],
                                         n_b))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_batches(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        mean_b = 0
        min_b = 1000000
        max_b = 0
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                max_ind = np.max(batches)
                batches_len = [np.sum(b < max_ind-0.5) for b in batches]

                for b_l in batches_len:
                    mean_b = 0.99 * mean_b + 0.01 * b_l
                max_b = max(max_b, np.max(batches_len))
                min_b = min(min_b, np.min(batches_len))

                print('{:d} < {:.1f} < {:d} /'.format(min_b, mean_b, max_b),
                      self.training_batch_limit,
                      batches_len)

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_neighbors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        hist_n = 500
        neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                for neighb_mat in neighbors:
                    print(neighb_mat.shape)

                counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                hists = [np.bincount(c, minlength=hist_n) for c in counts]

                neighb_hists += np.vstack(hists)

                print('***********************')
                dispstr = ''
                fmt_l = len(str(int(np.max(neighb_hists)))) + 1
                for neighb_hist in neighb_hists:
                    for v in neighb_hist:
                        dispstr += '{num:{fill}{width}}'.format(num=v, fill=' ', width=fmt_l)
                    dispstr += '\n'
                print(dispstr)
                print('***********************')

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_colors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                stacked_points = np_flat_inputs[:config.num_layers]
                stacked_colors = np_flat_inputs[-9]
                batches = np_flat_inputs[-7]
                stacked_labels = np_flat_inputs[-5]

                # Extract a point cloud and its color to save
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get points and colors (only for the concerned parts)
                    points = stacked_points[0][b]
                    colors = stacked_colors[b]
                    labels = stacked_labels[b]

                    write_ply('ScanNet_input_{:d}.ply'.format(b_i),
                              [points, colors[:, 1:4], labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'labels'])

                a = 1/0



                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = os.path.join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) +1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) +1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')

    # Preprocess methods
    # ------------------------------------------------------------------------------------------------------------------
    def preprocess_weaksup(self, verbose=True, data_path=None):
        import torch, os
        import numpy as np
        data_path = data_path if data_path is not None else self.data_path  # 'Data/ScanNet'
        f_list = ['points20', 'points50', 'points100', 'points200']
        for f in f_list:
            f = os.path.join(data_path, 'tasks/points', f)
            kw_arr = torch.load(f)
            np.savez(f, **kw_arr)
            # sanity check
            saved = dict(np.load(f'{f}.npz'))
            assert kw_arr.keys() == saved.keys()
            assert all(np.array_equal(kw_arr[k], saved[k]) for k in kw_arr.keys())
        return
