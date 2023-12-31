import os, re, gc, sys, time, pickle, psutil, subprocess
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

from config import log_config
from utils.logger import print_dict, print_table

# PLY reader
from utils.ply import read_ply, write_ply

FILE_DIR = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(FILE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'models'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'utils'))

from utils.tester import ModelTester
from utils.average_gradients import average_gradients
from utils.AdamWOptimizer import AdamWeightDecayOptimizer
from utils.logger import setup_logger
from utils.scheduler import StepScheduler, LrScheduler
from utils.metrics import AverageMeter
from utils.tf_graph_builder import GraphBuilder

DEBUG = False
class ModelTrainer:
    """
    get & train the model (potential multi-gpu training)
    """

    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.tester = ModelTester(config, verbose=False)

    def add_summary(self, model):
        with tf.variable_scope('summary'):
            summary = model.summary
            log_content = self.config.log_content

            if 'var' in log_content:
                summary['per_log'] += [tf.summary.histogram(v.name, v) for g, v in gvs]
            if 'gard' in log_content:
                summary['per_log'] += [tf.summary.histogram(f'{v.name}_grad', g) for g, v in gvs]

            sum_levels = ['per_step', 'per_log', 'per_epoch']
            assert all([k in sum_levels for k in summary.keys()]), f'undesired keys in summary dict: {str(summary.keys())}'
            for i in range(len(sum_levels)):
                summary[lv] = tf.summary.merge(summary[lv]) if summary[lv] else [tf.no_op]
            self.summary = summary
        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self):
        config = self.config
        with tf.Graph().as_default():  # use one graph

            # prepare compute graph
            g = GraphBuilder(config, verbose=self.verbose)
            ops, sess, grads, saver = g.ops, g.sess, g.grads, g.saver
            model, dataset = g.model, g.dataset
            self.model = model

            # printing model parameters
            if self.verbose:
                print('\n --------- printing grads {')
                re_list = ['.*bias:.*', '.*batch_normalization.*']  # skipping
                print_table([(v.name, g) for g, v in grads if not any([bool(re.fullmatch(expr, v.name)) for expr in re_list])], prefix='\t')
                print('} --------- printing grads')
                # all ops in graph
                print('\n --------- all ops {')
                re_list = ['optimizer.*', 'gpu_.*', 'gradients.*', 'save.*']  # '.*/batch_normalization/.*', '.*/bias:.*'  # skipping
                for n in tf.get_default_graph().as_graph_def().node:
                    if any([bool(re.fullmatch(expr, n.name)) for expr in re_list]): continue
                    print('\t', n.name)
                print('} --------- all ops')
                # model params
                all_params_size = sum([np.prod(v.shape) for _, v in grads])
                # all_params_size = tf.reduce_sum([tf.reduce_prod(v.shape) for _, v in grads])
                # all_params_size = sess.run(all_params_size)
                print(f'==> Model have {all_params_size} total Params', flush=True)

            # init sess
            sess.run(tf.global_variables_initializer())
            if self.config.model_path:
                except_list = [f'.*{n}.*' for n in self.config.exclude_vars] + ['optimizer.*'] if not self.config.continue_training else []
                g.restore(sess, self.config.model_path, except_list=except_list)
                print(f'Model restored -- {self.config.model_path}')

            # running voting - used throughout the training process (accumulated voting)
            validation_probs = self.tester.init_pointcloud_log(dataset, 'validation', config.num_classes)

            # train func
            if config.debug_nan:
                self.train_one_epoch = self.train_one_epoch_debug

            # train
            metric_best = None
            # save_snap = [i for i in range(1, config.max_epoch + 1) if i % config.save_freq == 0]
            lr_scheduler = LrScheduler(config)
            snap_path = os.path.join(config.saving_path, config.snap_dir, config.snap_prefix)
            for epoch in range(1, config.max_epoch + 1):
                print(f'\n****EPOCH {epoch}****')
                lr = lr_scheduler.learning_rate

                tic1 = time.time()
                step = self.train_one_epoch(sess, ops, epoch, lr, g=g)
                tic2 = time.time()
                print(f'total time: {(tic2 - tic1)/60:.1f}min, learning rate = {lr:.7f}', flush=True)

                if epoch % config.val_freq == 0:
                    metric = self.tester.val_running_vote(sess, ops, dataset, model, validation_probs)  # running voting
                    if metric_best is None or metric > metric_best:  # keep the best val
                        metric_best = metric
                        saver.save(sess, snap_path + '-best')
                        print('best saved')
                        # if config.save_best:
                        #     saver.save(sess, snap_path + '-best')
                        # if config.save_best == 'center':
                        #     epoch_start = max(epoch // config.save_freq - config.max_to_keep // 2, 1)
                        #     save_snap = [i * config.save_freq for i in range(epoch_start, epoch_start + config.max_to_keep + 1)]
                        #     save_snap = [i for i in save_snap if i != epoch]
                # if epoch in save_snap:
                if config.save_freq and epoch % config.save_freq == 0:
                    saver.save(sess, snap_path, global_step=epoch)
                lr_scheduler.step(epoch=1, step=step)

            # val & save last model if missed
            if epoch % config.val_freq != 0:
                self.tester.val_running_vote(sess, ops, dataset, model, validation_probs)
            if config.save_freq and epoch % config.save_freq != 0:
                saver.save(sess, snap_path, global_step=epoch)
            print('\nfinished\n', flush=True)
        return

    def train_one_epoch(self, sess, ops, epoch, lr, g=None):
        """
        One epoch training
        """
        config = self.config

        is_training = True
        batch_time = AverageMeter()
        loss_meter = {k: AverageMeter() for k in ops['loss_dict']}

        train_ops = {'train_op': ops['train_op'], 'loss_dict': ops['loss_dict']}
        feed_dict = {ops['is_training']: is_training, ops['learning_rate']: lr}
        sess.run(ops['train_init_op'])

        batch_idx = 0
        end = time.time()
        while True:
            try:
                rst = sess.run(train_ops, feed_dict=feed_dict)

                if (batch_idx + 1) % config.update_freq == 0:
                    for k, v in rst['loss_dict'].items():
                        loss_meter[k].update(v)
                    batch_time.update(time.time() - end)
                    end = time.time()

                if (batch_idx + 1) % config.print_freq == 0:
                    loss_str = ' '.join([f'{n}={meter.avg:<6.2f}' for n, meter in loss_meter.items()])
                    print(f'Step {batch_idx+1:08d} ' + loss_str + f' ---{batch_time.avg:5.3f} s/batch', flush=True)

                batch_idx += 1
            except tf.errors.OutOfRangeError:
                break
        return batch_idx

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def show_memory_usage(self, batch_to_feed):

            for l in range(self.config.num_layers):
                neighb_size = list(batch_to_feed[self.in_neighbors_f32[l]].shape)
                dist_size = neighb_size + [self.config.num_kernel_points, 3]
                dist_memory = np.prod(dist_size) * 4 * 1e-9
                in_feature_size = neighb_size + [self.config.first_features_dim * 2**l]
                in_feature_memory = np.prod(in_feature_size) * 4 * 1e-9
                out_feature_size = [neighb_size[0], self.config.num_kernel_points, self.config.first_features_dim * 2**(l+1)]
                out_feature_memory = np.prod(out_feature_size) * 4 * 1e-9

                print('Layer {:d} => {:.1f}GB {:.1f}GB {:.1f}GB'.format(l,
                                                                   dist_memory,
                                                                   in_feature_memory,
                                                                   out_feature_memory))
            print('************************************')

    def train_one_epoch_debug(self, sess, ops, epoch, lr, g=None):
        """
        One epoch training
        """
        config = self.config

        is_training = True
        batch_time = AverageMeter()
        loss_meter = {k: AverageMeter() for k in ops['loss_dict']}

        inputs = self.model.inputs
        inputs_flat = {k: v for k, v in inputs.items() if not isinstance(v, (list, dict))}
        train_ops = {'train_op': ops['train_op'], 'loss_dict': ops['loss_dict'], 'inputs': inputs_flat, 'result_dict': ops['result_dict']}
        assert_ops = inputs['assert_ops'] if 'assert_ops' in inputs and len(inputs['assert_ops']) > 0 else []
        feed_dict = {ops['is_training']: is_training, ops['learning_rate']: lr}
        sess.run(ops['train_init_op'])

        if config.debug_grads:
            assert g is not None  # [(g, v), ...]
            train_ops['grads'] = g.grads

        batch_idx = 0
        end = time.time()
        while True:
            try:
                with tf.control_dependencies(assert_ops):
                    rst = sess.run(train_ops, feed_dict=feed_dict)

                # NaN appears
                if config.debug_grads:
                    self.debug_grads_nan(sess, inputs, train_ops, rst)

                if any([np.isnan(v) for v in rst['loss_dict'].values()]):
                    self.debug_nan(sess, rst['inputs'], rst['result_dict'], rst['loss_dict'])
                    raise ArithmeticError(f'NaN encountered !!!')

                if (batch_idx + 1) % config.update_freq == 0:
                    for k, v in rst['loss_dict'].items():
                        loss_meter[k].update(v)
                    batch_time.update(time.time() - end)
                    end = time.time()

                if (batch_idx + 1) % config.print_freq == 0:
                    loss_str = ' '.join([f'{n}={meter.avg:<6.2f}' for n, meter in loss_meter.items()])
                    print(f'Step {batch_idx+1:08d} ' + loss_str + f' ---{batch_time.avg:5.3f} s/batch', flush=True)

                batch_idx += 1
            except tf.errors.OutOfRangeError:
                break
        return batch_idx

    def debug_grads_nan(self, sess, inputs, ops, rst):
        grads = ops['grads']
        grads_v = rst['grads']

        nan_grads = [(g, v, g_val, v_val) for (g, v), (g_val, v_val) in zip(grads, grads_v) if np.isnan(g_val).any() or np.isnan(v_val).any()]
        if not nan_grads:
            return

        lines = []
        for g, v, g_val, v_val in nan_grads:
            g_nan = 100 * np.sum(np.isnan(g_val)) / np.prod(g_val.shape)
            v_nan = 100 * np.sum(np.isnan(v_val)) / np.prod(v_val.shape)
            lines.append([v.name, g, '-', v_val.shape, f'/ {v_nan:.1f}', 'val nan', g_val.shape, f'/ {g_nan:.1f}', 'grad nan'])
        print_table(lines)

        self.debug_nan(sess, rst['inputs'], rst['result_dict'], rst['loss_dict'])
        raise ArithmeticError(f'NaN encountered in grads checking !!!')
        return

    def debug_nan(self, sess, inputs, result_dict, loss_dict):
        """
        NaN happened, find where
        """

        print('\n\n------------------------ NaN DEBUG ------------------------\n')

        print('loss_dict :')
        print('*******************\n')
        print_dict(loss_dict)

        # Then print a list of the trainable variables and if they have nan
        print('List of variables :')
        print('*******************\n')
        all_vars = sess.run(tf.global_variables())
        for v, value in zip(tf.global_variables(), all_vars):
            nan_percentage = 100 * np.sum(np.isnan(value)) / np.prod(value.shape)
            line = v.name + (f'\t => {nan_percentage:.1f}% of values are NaN' if np.isnan(value).any() else '')
            print(line)

        print('Inputs :')
        print('********')

        #Print inputs
        for layer in range(self.config.num_layers):

            print(f'Layer : {layer}')

            points = inputs['points'][layer]
            neighbors = inputs['neighbors'][layer]
            pools = inputs['pools'][layer]
            upsamples = inputs['upsamples'][layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / np.prod(pools.shape)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / np.prod(upsamples.shape)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        features = inputs['features']
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        batch_weights = inputs['batch_weights']
        in_batches = inputs['in_batches']
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        out_batches = inputs['out_batches']
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        point_labels = inputs['point_labels']
        if self.config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs['object_labels']
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
        augment_scales = inputs['augment_scales']
        augment_rotations = inputs['augment_rotations']

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        for layer in range(self.config.num_layers):

            print(f'\nLayer : {layer}')

            neighbors = inputs['neighbors'][layer]
            pools = inputs['pools'][layer]
            upsamples = inputs['upsamples'][layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            max_n = np.max(pools)
            nums = np.sum(pools < max_n - 0.5, axis=-1)
            print('min pools =>', np.min(nums))

            max_n = np.max(upsamples)
            nums = np.sum(upsamples < max_n - 0.5, axis=-1)
            print('min upsamples =>', np.min(nums))


        print('\n--- NaN Debug Print End ---\n\n', flush=True)

        # # save everything to reproduce error - inputs/logits
        # file1 = os.path.join(self.config.saving_path, 'all_debug_inputs.pkl')
        # with open(file1, 'wb') as f1:
        #     pickle.dump(inputs, f1)
        # file1 = os.path.join(self.config.saving_path, 'all_debug_logits.pkl')
        # with open(file1, 'wb') as f1:
        #     pickle.dump(logits, f1)


        time.sleep(0.5)

