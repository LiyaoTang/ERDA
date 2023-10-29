# Basic libs
import os, gc, re, sys, time, json
from functools import partial
ROOT_DIR = os.path.abspath(os.path.join(__file__, '../', '../'))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

from sklearn.neighbors import KDTree
from collections import defaultdict

# PLY reader
from utils.ply import read_ply, write_ply

# Helper
from utils.storage import *
from utils.logger import log_percentage, print_dict, print_mem

# Metrics
from utils.metrics import AverageMeter, Metrics, metrics_from_confusions, metrics_from_result
from sklearn.metrics import confusion_matrix


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose

        self.save_extra = {}  # for saving with extra ops

        if config.dataset in ['S3DIS', 'ScanNet', 'SensatUrban']:
            self.val_running_vote = self.val_running_vote_seg
            self.val_vote = self.val_vote_seg
            self.test_vote = self.test_vote_seg
        else:
            raise NotImplementedError(f'not supported dataset: {config.dataset}')

    def init_pointcloud_log(self, dataset, split, d, dtype=np.float32, init_fn=np.zeros):
        shape = lambda l: [l, d] if d else [l]  # d - size of last dimension => each point d-dim [N, d] (d = None to have [N])
        log = [init_fn(shape=shape(t.data.shape[0]), dtype=dtype) for t in dataset.input_trees[split]]
        return log

    def initialize(self, ops, dataset, model, split):
        # initialize cum_dict & ops
        config = self.config
        ncls = config.num_classes

        run_ops = {k: ops['result_dict'][k] for k in ['inputs', 'seg']}  # assumes per-gpu rst - support multi-gpu
        cum_dict = {
            'prob': self.init_pointcloud_log(dataset, split, ncls)
        }

        extra_ops = [k for k in config.extra_ops.split('-') if k]
        extra_ops_solved = extra_ops.copy()
        for k in extra_ops:
            if k in ['prob', 'conf']:
                continue
            else:
                raise ValueError(f'not supported extra ops k = {k} from {config.extra_ops}')

        return run_ops, cum_dict, extra_ops_solved

    # Val methods
    # ------------------------------------------------------------------------------------------------------------------

    def val_running_vote_seg(self, sess, ops, dataset, model, validation_probs, epoch=1):
        """
        One epoch validating - running voting used during training, main task results only
        """

        val_smooth = 0.95  # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)

        result_dict = {k: ops['result_dict'][k] for k in ['inputs', 'seg']}  # result dict for seg
        val_ops = {'loss_dict': ops['loss_dict'], 'result_dict': result_dict}
        feed_dict = {ops['is_training']: False}

        # Initialise iterator
        sess.run(ops['val_init_op'])

        ep = 0
        loss_meter = {k: AverageMeter() for k in val_ops['loss_dict']} if 'loss_dict' in val_ops else{}
        cum_dict = {
            'conf': 0,  # conf from current validation
            'prob': validation_probs,  # accumulating probs
        }
        while ep < epoch:
            try:
                rst = sess.run(val_ops, feed_dict=feed_dict)

                loss_dict = rst['loss_dict'] if 'loss_dict' in rst else {}
                cur_rst = rst['result_dict']  # per-gpu result

                for k, v in loss_dict.items():
                    loss_meter[k].update(v)

                # Stack all validation predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=val_smooth)

            except tf.errors.OutOfRangeError:
                ep += 1
                pass

        if loss_meter:
            print(f'val loss avg:', ' '.join([f'{loss_n} = {meter.avg:.3f}' for loss_n, meter in loss_meter.items()]))

        label_to_idx = dataset.label_to_idx
        proportions = dataset.val_proportions
        cur_m = metrics_from_confusions(cum_dict['conf'], proportions=proportions)  # use sampled pred-label of current epoch
        vote_m = metrics_from_result(validation_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)  # use the accumulated per-point voting

        print(f'metrics - current     {cur_m}\n'
              f'        - accumulated {vote_m}', flush=True)
        return cur_m


    def val_vote_seg(self, sess, ops, dataset, model, num_votes=20):
        """
        Voting validating
        """

        feed_dict = {ops['is_training']: False}

        # Smoothing parameter for votes
        val_smooth = 0.95

        # Initialise iterator with val data
        sess.run(ops['val_init_op'])

        # Initiate global prediction over val clouds
        label_to_idx = dataset.label_to_idx
        proportions = dataset.val_proportions
        val_ops, cum_dict, extra_ops = self.initialize(ops, dataset, model, 'validation')
        val_probs = cum_dict['prob']

        vote_ind = 0
        last_min = -0.5
        if self.config.debug:
            print_dict(val_ops, head='val_vote_seg - val_ops')
        while last_min < num_votes:
            try:
                cur_rst = sess.run(val_ops, feed_dict=feed_dict)
                # Stack all validation predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=val_smooth)

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_potentials['validation'])
                if self.verbose:
                    print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)
                if last_min + 1 < new_min:
                    # Update last_min
                    last_min += 1

                    if self.verbose > 1:
                        # Show vote results on subcloud (match original label to valid) => not the good values here
                        vote_m = metrics_from_result(val_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)
                        print('==> Confusion on sub clouds: ', vote_m.scalar_str)

                    if self.verbose > 1 and int(np.ceil(new_min)) % 2 == 0:
                        # Project predictions
                        vote_m = metrics_from_result(val_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, projections=dataset.validation_proj)
                        print('==> Confusion on full clouds:', vote_m)

                sess.run(ops['val_init_op'])
                vote_ind += 1

        vote_m = metrics_from_result(val_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)
        print('==> Confusion on sub clouds - final: ', vote_m.scalar_str)

        # Project predictions
        print('==> Confusion on full clouds - final:')
        vote_m = metrics_from_result(val_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, projections=dataset.validation_proj)
        vote_m.print()
        print('\nfinished\n', flush=True)

        return


    # Test methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_classification(self, model, dataset, num_votes=100):

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate votes
        average_probs = np.zeros((len(dataset.input_labels['test']), nc_model))
        average_counts = np.zeros((len(dataset.input_labels['test']), nc_model))

        mean_dt = np.zeros(2)
        last_display = time.time()
        while np.min(average_counts) < num_votes:

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            probs = []
            targets = []
            obj_inds = []
            count = 0

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits, model.labels, model.inputs['object_inds'])
                    prob, labels, inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Get probs and labels
                    probs += [prob]
                    targets += [labels]
                    obj_inds += [inds]
                    count += prob.shape[0]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > self.gap_display:
                        last_display = t[-1]
                        message = 'Vote {:.0f} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(np.min(average_counts),
                                             100 * count / dataset.num_test,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Average votes
            # *************

            # Stack all validation predictions
            probs = np.vstack(probs)
            targets = np.hstack(targets)
            obj_inds = np.hstack(obj_inds)

            if np.any(dataset.input_labels['test'][obj_inds] != targets):
                raise ValueError('wrong object indices')

            # Compute incremental average (predictions are always ordered)
            average_counts[obj_inds] += 1
            average_probs[obj_inds] += (probs - average_probs[obj_inds]) / (average_counts[obj_inds])

            # Save/Display temporary results
            # ******************************

            test_labels = np.array(dataset.label_values)

            # Compute classification results
            C1 = confusion_matrix(dataset.input_labels['test'],
                                  np.argmax(average_probs, axis=1),
                                  test_labels)

            ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
            print('Test Accuracy = {:.1f}%'.format(ACC))

            s = ''
            for cc in C1:
                for c in cc:
                    s += '{:d} '.format(c)
                s += '\n'
            print(s)



            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return

    def test_multi_segmentation(self, model, dataset, num_votes=100, num_saves=10):

        ##################
        # Pre-computations
        ##################

        print('Preparing test structures')
        t1 = time.time()

        # Collect original test file names
        original_path = join(dataset.path, 'test_ply')
        test_names = [f[:-4] for f in listdir(original_path) if f[-4:] == '.ply']
        test_names = np.sort(test_names)

        original_labels = []
        original_points = []
        projection_inds = []
        for i, cloud_name in enumerate(test_names):

            # Read data in ply file
            data = read_ply(join(original_path, cloud_name + '.ply'))
            points = np.vstack((data['x'], -data['z'], data['y'])).T
            original_labels += [data['label'] - 1]
            original_points += [points]

            # Create tree structure to compute neighbors
            tree = KDTree(dataset.input_points['test'][i])
            projection_inds += [np.squeeze(tree.query(points, return_distance=False))]

        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))

        ##########
        # Initiate
        ##########

        # Test saving path
        if config.save_test:
            test_path = join(model.saving_path, 'test')
            if not exists(test_path):
                makedirs(test_path)
        else:
            test_path = None

        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)

        # Initiate result containers
        average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]

        #####################
        # Network predictions
        #####################

        mean_dt = np.zeros(2)
        last_display = time.time()
        for v in range(num_votes):

            # Run model on all test examples
            # ******************************

            # Initiate result containers
            all_predictions = []
            all_obj_inds = []

            while True:
                try:

                    # Run one step of the model
                    t = [time.time()]
                    ops = (self.prob_logits,
                           model.labels,
                           model.inputs['super_labels'],
                           model.inputs['object_inds'],
                           model.inputs['in_batches'])
                    preds, labels, obj_labels, o_inds, batches = self.sess.run(ops, {model.dropout_prob: 1.0})
                    t += [time.time()]

                    # Stack all predictions for each class separately
                    max_ind = np.max(batches)
                    for b_i, b in enumerate(batches):

                        # Eliminate shadow indices
                        b = b[b < max_ind - 0.5]

                        # Get prediction (only for the concerned parts)
                        obj = obj_labels[b[0]]
                        predictions = preds[b][:, :config.num_classes[obj]]

                        # Stack all results
                        all_predictions += [predictions]
                        all_obj_inds += [o_inds[b_i]]

                    # Average timing
                    t += [time.time()]
                    mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Display
                    if (t[-1] - last_display) > self.gap_display:
                        last_display = t[-1]
                        message = 'Vote {:d} : {:.1f}% (timings : {:4.2f} {:4.2f})'
                        print(message.format(v,
                                             100 * len(all_predictions) / dataset.num_test,
                                             1000 * (mean_dt[0]),
                                             1000 * (mean_dt[1])))

                except tf.errors.OutOfRangeError:
                    break

            # Project predictions on original point clouds
            # ********************************************

            print('\nGetting test confusions')
            t1 = time.time()

            for i, probs in enumerate(all_predictions):

                # Interpolate prediction from current positions to original points
                obj_i = all_obj_inds[i]
                proj_predictions = probs[projection_inds[obj_i]]

                # Average prediction across votes
                average_predictions[obj_i] = average_predictions[obj_i] + \
                                             (proj_predictions - average_predictions[obj_i]) / (v + 1)

            Confs = []
            for obj_i, avg_probs in enumerate(average_predictions):

                # Compute confusion matrices
                parts = [j for j in range(avg_probs.shape[1])]
                Confs += [confusion_matrix(original_labels[obj_i], np.argmax(avg_probs, axis=1), parts)]


            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Save the best/worst segmentations per class
            # *******************************************

            print('Saving test examples')
            t1 = time.time()

            # Regroup confusions per object class
            Confs = np.array(Confs)
            obj_mIoUs = []
            for l in dataset.label_values:

                # Get confusions for this object
                obj_inds = np.where(dataset.input_labels['test'] == l)[0]
                obj_confs = np.stack(Confs[obj_inds])

                # Get IoU
                obj_IoUs = IoU_from_confusions(obj_confs)
                obj_mIoUs += [np.mean(obj_IoUs, axis=-1)]

                # Get X best and worst prediction
                order = np.argsort(obj_mIoUs[-1])
                worst_inds = obj_inds[order[:num_saves]]
                best_inds = obj_inds[order[:-num_saves-1:-1]]
                worst_IoUs = obj_IoUs[order[:num_saves]]
                best_IoUs = obj_IoUs[order[:-num_saves-1:-1]]

                # Save the names in a file
                if config.save_test:
                    obj_path = join(test_path, dataset.label_to_names[l])
                    if not exists(obj_path):
                        makedirs(obj_path)
                    worst_file = join(obj_path, 'worst_inds.txt')
                    best_file = join(obj_path, 'best_inds.txt')
                    with open(worst_file, "w") as text_file:
                        for w_i, w_IoUs in zip(worst_inds, worst_IoUs):
                            text_file.write('{:d} {:s} :'.format(w_i, test_names[w_i]))
                            for IoU in w_IoUs:
                                text_file.write(' {:.1f}'.format(100*IoU))
                            text_file.write('\n')

                    with open(best_file, "w") as text_file:
                        for b_i, b_IoUs in zip(best_inds, best_IoUs):
                            text_file.write('{:d} {:s} :'.format(b_i, test_names[b_i]))
                            for IoU in b_IoUs:
                                text_file.write(' {:.1f}'.format(100*IoU))
                            text_file.write('\n')

                    # Save the clouds
                    for i, w_i in enumerate(worst_inds):
                        filename = join(obj_path, 'worst_{:02d}.ply'.format(i+1))
                        preds = np.argmax(average_predictions[w_i], axis=1).astype(np.int32)
                        write_ply(filename,
                                [original_points[w_i], original_labels[w_i], preds],
                                ['x', 'y', 'z', 'gt', 'pre'])

                    for i, b_i in enumerate(best_inds):
                        filename = join(obj_path, 'best_{:02d}.ply'.format(i+1))
                        preds = np.argmax(average_predictions[b_i], axis=1).astype(np.int32)
                        write_ply(filename,
                                [original_points[b_i], original_labels[b_i], preds],
                                ['x', 'y', 'z', 'gt', 'pre'])

            t2 = time.time()
            print('Done in {:.1f} s\n'.format(t2 - t1))

            # Display results
            # ***************

            objs_average = [np.mean(mIoUs) for mIoUs in obj_mIoUs]
            instance_average = np.mean(np.hstack(obj_mIoUs))
            class_average = np.mean(objs_average)

            print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
            print('-----|------|--------------------------------------------------------------------------------')

            s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
            for AmIoU in objs_average:
                s += '{:4.1f} '.format(100 * AmIoU)
            print(s + '\n')

            # Initialise iterator with test data
            self.sess.run(dataset.test_init_op)

        return

    def test_vote_seg(self, sess, ops, dataset, model, num_votes=20, test_path=None, make_zip=True):

        config = self.config
        assert os.path.isdir(config.saving_path), f'not a dir: {config.saving_path}'
        if test_path is None:
            test_path = os.path.join(config.saving_path, 'test')
        os.makedirs(test_path, exist_ok=True)

        options = None  # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = None  # tf.RunMetadata()
        feed_dict = {ops['is_training']: False}

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with test data
        sess.run(ops['test_init_op'])

        # Initiate global prediction over val clouds
        test_ops, cum_dict, extra_ops = self.initialize(ops, dataset, model, 'test')
        test_probs = cum_dict['prob']

        vote_ind = 0
        last_min = -0.5 
        if config.num_votes:
            num_votes = config.num_votes
        while last_min < num_votes:
            try:
                cur_rst = sess.run(test_ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                # Stack all test predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=test_smooth)

            except tf.errors.OutOfRangeError:
                # NOTE: need to check
                new_min = np.min(dataset.min_potentials['test'])
                if self.verbose:
                    print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)

                if last_min + 1 < new_min:
                    # Update last_min
                    last_min += 1

                # if int(last_min) > 0 and int(last_min) // 5 == 0:  # periodic test results
                #     self.project_test_predictions(dataset, test_path)

                sess.run(ops['test_init_op'])
                vote_ind += 1

        if self.verbose:
            new_min = np.min(dataset.min_potentials['test'])
            print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)

        self.project_test_predictions(dataset, test_probs, test_path)
        print('\nfinished\n', flush=True)

        if make_zip:
            zip_name = test_path.split(os.sep)  # cfg name / Log_* / test_*
            zip_name = '_'.join([i for i in ['test', *zip_name[-3:-1], zip_name[-1][len('test'):].strip('_')] if i])
            # include test_* dir (except Semantic3D, ScanNet)
            j = 'j' if config.dataset in ['ScanNet', 'Semantic3D', 'SensatUrban'] else ''
            os.system(f'cd {os.path.dirname(test_path)}; zip -rmTq{j} {zip_name}.zip {test_path.split(os.sep)[-1]}/*')  # -m to move, -j junk file, -T test integrity, -q quiet
            os.system(f'rm -r {test_path}')
        return

    def project_test_predictions(self, dataset, test_probs, test_path):

        # Project predictions
        t1 = time.time()
        files = dataset.test_files
        ignored_inds = None
        if hasattr(dataset, 'ignored_labels_test'):
            ignored_inds = dataset.label_to_idx[[l for l in dataset.ignored_labels_test if l not in dataset.ignored_labels]].astype(int)

        config = self.config
        if config.save_test:
            pred_path = os.sep.join([*test_path.split(os.sep)[:-1], test_path.split(os.sep)[-1].replace('test', 'predictions')])  # model pred
            os.makedirs(pred_path, exist_ok=True)

        for i_test, file_path in enumerate(files):

            # Reproject probs
            probs = test_probs[i_test][dataset.test_proj[i_test], :]

            # Remove invalid classes in test
            if ignored_inds is not None:
                probs[:, ignored_inds] = 0

            # Get the predicted labels
            preds = dataset.idx_to_label[np.argmax(probs, axis=-1)]

            # Save plys - predictions & probs
            cloud_name = file_path.split('/')[-1]
            if config.save_test:
                points = dataset.load_evaluation_points(file_path)  # test original points
                pots = dataset.potentials['test'][i_test][dataset.test_proj[i_test]]  # project potentials on original points
                test_name = os.path.join(pred_path, cloud_name)
                prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values if label not in dataset.ignored_labels]
                write_ply(test_name,
                        [points, preds, pots, probs],
                        ['x', 'y', 'z', 'preds', 'pots'] + prob_names)

            # Save ascii preds - submission files
            if config.dataset == 'Semantic3D':
                ascii_name = os.path.join(test_path, dataset.ascii_files[cloud_name])
                np.savetxt(ascii_name, preds, fmt='%d')
            elif config.dataset == 'SensatUrban':
                ascii_name = os.path.join(test_path, f'{cloud_name[:-4]}.label')
                preds.astype(np.uint8).tofile(ascii_name)
            else:
                ascii_name = os.path.join(test_path, cloud_name[:-4] + '.txt')
                np.savetxt(ascii_name, preds, fmt='%d')

        t2 = time.time()
        if self.verbose:
            print('\nReproject Vote in {:.1f}s\n'.format(t2-t1))


    # Utilities
    # ------------------------------------------------------------------------------------------------------------------

    def cumulate_probs(self, dataset, model, rst, cum_dict, task, smooth):
        # cum_dict - {cum_dict name : {args : rst_dict}}

        # iterate over gpu
        for gpu_i, cloud_inds in enumerate(rst['inputs']['cloud_inds']):
            point_inds = rst['inputs']['point_inds'][gpu_i]

            b_start = 0
            # iterate over clouds
            for b_i, c_i in enumerate(cloud_inds):  # [B]
                if 'batches_len' in rst['inputs']:  # [BxN] - stacked
                    b_len = rst['inputs']['batches_len'][gpu_i][0][b_i]  # npoints in cloud
                    b_i = np.arange(b_start, b_start + b_len)
                    b_start += b_len
                else:  # [B, N] - batched
                    pass
                inds = point_inds[b_i]  # input point inds

                probs = rst[task]['probs'][gpu_i][b_i]
                labels = rst[task]['labels'][gpu_i][b_i]
                if np.all(labels == -1):
                    # c_pts = np.array(dataset.input_trees['validation'][c_i].data, copy=False)[inds].mean(axis=0)
                    # unique_l_cnt = np.unique(dataset.input_labels['validation'][c_i][inds], return_counts=True)
                    # raise ValueError(f'all invalid labels found in cumulate_prob: cloud_inds={c_i}, center_pts={c_pts}'
                    #                 f'input_labels & counts - {unique_l_cnt}')
                    continue
                if 'conf' in cum_dict:
                    cur_conf = confusion_matrix(labels, np.argmax(probs, axis=-1).astype(np.int), labels=np.arange(dataset.num_classes))
                    cum_dict['conf'] += cur_conf
                if 'prob' in cum_dict:
                    cum_dict['prob'][c_i][inds] = smooth * cum_dict['prob'][c_i][inds] + (1 - smooth) * probs
                if 'feature' in cum_dict:
                    cum_dict['feature'][c_i][inds] = smooth * cum_dict['feature'][c_i][inds] + (1 - smooth) * rst[task]['latent'][gpu_i][b_i]

    def _search_func(self, k_r, cloud_idx, split, dataset, neighbor_dict, verbose=True):  # create tf_ops of generating neighbor_idx & get result
        if cloud_idx in neighbor_dict[k_r]:
            return neighbor_dict[k_r][cloud_idx]

        config = self.config
        points = np.array(dataset.input_trees[split][cloud_idx].data, copy=False)  # [N, 3]

        from ops import get_tf_func
        func = get_tf_func(config.search, verbose=verbose)

        if config.search in ['knn']:
            tf_ops = tf.squeeze(func(points[None, ...], points[None, ...], k_r), axis=0)
        elif config.search in ['radius']:
            tf_ops = func(points, points, [len(points)], [len(points)], k_r)
            # if hasattr(dataset, 'neighborhood_limits'):
            #     print('neighborhood_limits', dataset.neighborhood_limits[0])
            #     tf_ops = tf_ops[..., :dataset.neighborhood_limits[0]]
        else:
            raise

        if verbose:
            print_mem(f'k = {k_r} - start', check_time=True, check_sys=True, flush=True)
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)) as s:
            neighbor_idx = s.run(tf_ops)
        if verbose:
            print_mem(f'neighbor_idx {neighbor_idx.shape}', check_time=True, check_sys=True, flush=True)

        neighbor_dict[k_r][cloud_idx] = neighbor_idx  # neighbor idx - np arr
        return neighbor_idx
