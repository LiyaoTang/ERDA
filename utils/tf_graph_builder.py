import os, re, sys, psutil, inspect
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import models, datasets
from utils.tf_utils import restore
from utils.logger import print_dict, print_mem
from utils.storage import dict_list
from utils.average_gradients import average_gradients
from collections import defaultdict

class GraphBuilder(object):

    def __init__(self, config, graph=None, verbose=True):
        """
        get the full compute graph including dataset, model inference, loss, optimizer, lr scheduler and required ops
        """

        if graph is not None:  # if graph specified
            with graph.as_default():
                return self.__init__(config, None, verbose)

        if isinstance(config.rand_seed, int):  # set seed
            tf.set_random_seed(config.rand_seed)
            np.random.seed(config.rand_seed)
        if verbose:
            print(f'==> np random seed = {np.random.get_state()[1][0]}')

        # model & dataset fn
        self.get_dataset = getattr(datasets, f'{config.dataset}Dataset')  # datasets.[name]Dataset
        self.get_model = models.get_model
        # if config.distribute == 'tf_device':  # full compute graph (handle devices & platforms)
        #     self.build = self.build_devices
        # else:
        #     raise NotImplementedError(f'not supported type of distributing graphs: config.distribute={config.distribute}')

        # Get dataset
        if verbose:
            print('==> Preparing datasets...')
        dataset = self.get_dataset(config, verbose)
        dataset.initialize(verbose)
        if verbose:
            print('==> setting dataset info:')
            print_dict(dataset.info, prefix='\t')
            print_mem('>>> dataset built')
        config.update(dataset.info)

        # placeholder
        is_training = tf.placeholder(tf.bool, shape=())
        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        # learning_rate = tf.get_variable('learning_rate', [], initializer=tf.constant_initializer(float('nan')), trainable=False)

        # # build model
        # grads, total_loss_dict, total_result_dict, model = self.build(dataset, is_training, config, verbose=verbose)

        # -------------------------------------------
        # Get model and loss on multiple GPU devices
        # -------------------------------------------
        # Allocating variables on CPU first will greatly accelerate multi-gpu training.
        # Ref: https://github.com/kuza55/keras-extras/issues/21
        flat_inputs = dataset.flat_inputs
        if config.cpu_variables:
            self.get_model(flat_inputs[0], is_training, config=config, verbose=verbose)
        tower_grads = []
        total_losses = []
        total_result = []
        for igpu in range(config.gpu_num):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if config.cpu_variables else tf.AUTO_REUSE):
                name_scope = f'gpu_{igpu}' if config.cpu_variables or igpu > 0 else ''
                verbose = not bool(name_scope)
                with tf.device(f'/gpu:{igpu}'), tf.name_scope(name_scope) as scope:
                    flat_inputs_i = flat_inputs[igpu]
                    model = self.get_model(flat_inputs_i, is_training, config=config, scope=scope, verbose=verbose)  # inference model

                    # collect per-gpu info
                    result_dict = model.get_result()  # inference result
                    total_result.append(result_dict)

                    loss_dict = model.get_loss()  # loss
                    total_losses.append(loss_dict)

                    var_list = tf.trainable_variables()  # vars & grads
                    var_list = self.collect_vars(var_list, include_k=config.vars_train, except_k=config.vars_freeze)
                    grads = tf.gradients(loss_dict['loss'], var_list, colocate_gradients_with_ops=config.colocate_gradients_with_ops)  # normally, should NOT co-locate
                    grads = list(zip(grads, var_list))
                    tower_grads.append(grads)
        total_inputs = dict_list(flat_inputs)
        total_result = dict_list(total_result)
        total_losses = dict_list(total_losses)

        # average losses from multiple GPUs
        with tf.variable_scope('losses'):
            total_losses = {k: tf.reduce_mean(v, name=k) if len(v) > 1 else v[0] for k, v in total_losses.items()}

        # average grad
        with tf.variable_scope('gradients'):
            # [(gradient, variable), ...] - gradient averaged over gpu towers (if >1)
            grads = average_gradients(tower_grads, grad_norm=config.grad_norm, raise_on_none=config.grad_raise_none, grad_reduce=config.grad_reduce)

        # setup optimizer
        with tf.variable_scope('optimizer'):
            if config.optimizer == 'sgd':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)
            elif config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif config.optimizer == 'adamW':
                from utils.AdamWOptimizer import AdamWeightDecayOptimizer
                optimizer = AdamWeightDecayOptimizer(learning_rate=learning_rate, weight_decay_rate=config.weight_decay, exclude_from_weight_decay=["bias"])

            # if config.mixed_precision:
            #     optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

            # momentume as update ops
            update_ops = self.get_momentum_update(model, config, total_inputs, total_result)
            for ops in update_ops:  # add to collection
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ops)

            # train op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads)
            # train_op = optimizer.apply_gradients(grads)
            # train_op = tf.group([train_op, update_ops])

        # saver
        save_vars = None
        if config.save_compact:
            save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
            if isinstance(config.save_compact, bool):
                pass
            elif isinstance(config.save_compact, str) and config.save_compact == 'trained':
                vars_grads = {v: g for g, v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')}
                save_vars = [v for v in save_vars if v in vars_grads and vars_grads[v] is not None]  # save only trained
            else:
                raise ValueError(f'not support save_compact={config.save_compact}')
        saver = tf.train.Saver(save_vars, max_to_keep=int(config.max_to_keep))

        # summary
        with tf.variable_scope('summary'):
            if config.summary and isinstance(config.summary, str):
                inputs = model.inputs
                if 'summary' not in inputs:
                    inputs['summary'] = defaultdict(lambda: [])
            if config.summary == 'loss':
                inputs['summary']['per_step'] += [tf.summary.scalar(k, v) for k, v in total_losses.items()]
            # log grads - debug use
            # inputs = model.inputs
            # inputs['summary'] = defaultdict(lambda: [])
            # from models.utils import tf_Print
            # for i, (g, v) in enumerate(grads):
            #     if config.summary:
            #         inputs['summary']['per_step'] += [tf.summary.histogram(f'{v.name}/v', v)]
            #         inputs['summary']['per_step'] += [tf.summary.histogram(f'{v.name}/g', g)]
            #     if v.name in [
            #             'model/resnet_scene_segmentation_head/up_conv3/weights:0',
            #             'model/resnet_scene_segmentation_head/segmentation_head/weights:0',
            #     ]:
            #         print(f'print grad - {v.name}')
            #         g = tf_Print(g, [f'grads - {v.name}', g])
            #         grads[i] = (g, v)
            # input('\nprint above grads')
        # summary - merge
        summary_dict = {}  # {level : merged op}
        if config.summary:
            sum_levels = ['per_step', 'per_log', 'per_epoch']
            summary_ops = model.inputs['summary'] if 'summary' in model.inputs else {k: [] for k in sum_levels}
            assert all([k in sum_levels for k in summary_ops]), f'undesired keys in summary ops: {summary_ops.keys()}'
            for i in range(len(sum_levels)):
                lv = sum_levels[-i - 1]
                ops = sum([summary_ops[k] for k in sum_levels[:len(sum_levels)-i]], [])
                summary_dict[lv] = tf.summary.merge(ops) if len(ops) > 0 else tf.no_op()

        # Create a session
        cProto = tf.ConfigProto()
        if config.gpu_allow_growth:
            cProto.gpu_options.allow_growth = True
        if config.debug_single:
            cProto.device_count['CPU'] = 1
        # config.intra_op_parallelism_threads = config.inter_op_parallelism_threads = psutil.cpu_count(logical=False)  # set to num of physical (default to logical) cpu cores
        cProto.allow_soft_placement = bool(config.allow_soft_placement) or not bool(config.gpu_devices)  # if specified or cpu-only
        cProto.log_device_placement = False
        sess = tf.Session(config=cProto)

        ops = {
            'train_init_op': dataset.train_init_op,
            'val_init_op': dataset.val_init_op,
            'test_init_op': dataset.test_init_op,

            'train_op': train_op,
            'is_training': is_training,
            'learning_rate': learning_rate,

            'inputs': dict(total_inputs),
            'loss_dict': dict(total_losses),
            'result_dict': dict(total_result),
            'summary_dict': dict(summary_dict),
        }
        if verbose:
            print_mem('>>> model built')
            print('\n -------- inputs {')
            print_dict(model.inputs, prefix='\t')
            print('} --------- inputs')
            print('\n -------- loss_dict {')
            print_dict(total_losses, prefix='\t')
            print('} --------- loss_dict')
            print('\n -------- result_dict {')
            print_dict(total_result, prefix='\t')
            print('} --------- result_dict')

        self.ops = ops
        self.sess = sess
        self.grads = grads
        self.saver = saver

        self.model = model
        self.dataset = dataset

    # -------------------------------------------
    # Other utils & interfaces
    # -------------------------------------------

    def collect_vars(self, var_list, include_k=[], except_k=[], match='search'):
        # collect specified vars - default to all vars
        var_collect = []
        match_func = getattr(re, match)
        include_k = [include_k] if include_k and isinstance(include_k, str) else include_k
        except_k = [include_k] if except_k and isinstance(except_k, str) else except_k
        for v in var_list:
            if include_k and not any(match_func(k, v.name) for k in include_k):
                continue
            if except_k and any(match_func(k, v.name) for k in except_k):
                continue
            var_collect.append(v)
        return var_collect

    def get_momentum_update(self, model, config, total_inputs, total_result):
        # collect update ops for momentum update
        update_ops = []

        # update ops - momentum dict
        # NOTE - can be done in per-head fashion
        # => check only sepcial 'momentum_update_stage'
        for head_n, head_d in total_result.items():
            if 'momentum_dict' not in head_d or 'momentum_dict' not in total_inputs: continue
            if head_n not in total_inputs['momentum_dict']:
                raise KeyError(f'building momentum cycle for head {head_n}: missing tensor for momentum dict')
            head_cfg = model.head_dict['config'][head_n]

            # per-device input/output
            mom_in = total_inputs['momentum_dict'][head_n]  # {k : [v = tensor]}, with inputs['momentum_dict'] = {head_n: {k : placeholder/vars}}
            mom_out = head_d['momentum_dict']  # {k: [v = tensor]}
            for k, v_out in mom_out.items():
                v_in = mom_in[k]

                # collect for update
                mom_avg = head_cfg.momentum_update
                mom_avg = float(mom_avg) if isinstance(mom_avg, (str, int)) else mom_avg  # can be variable
                with tf.variable_scope(f'mom_dict_update/{head_n}/{k}'):
                    if head_cfg.momentum_update_stage == 'glb_avg':
                        # average over devices
                        v_out = tf.reduce_mean(tf.stack(v_out, axis=0), axis=0)
                        v_out = [v_in[i] * mom_avg + v_out * (1 - mom_avg) for i in range(config.gpu_num)]

                    elif head_cfg.momentum_update_stage == 'glb_sum':
                        # sum over devices
                        v_out = tf.reduce_sum(tf.stack(v_out, axis=0), axis=0)
                        v_out = [v_in[i] * mom_avg + v_out * (1 - mom_avg) for i in range(config.gpu_num)]

                # create update ops
                for igpu in range(config.gpu_num):  # assign to each device input
                    with tf.variable_scope(f'gpu_{igpu}/mom_dict_update/{head_n}/{k}', reuse=True):
                        update_ops += [tf.assign(v_in[igpu], v_out[igpu])]

        return update_ops



    def restore(self, *args, **kwargs):
        argspec = inspect.getfullargspec(restore)
        kwargs.update(zip(argspec.args, args))
        kw_self = {'session': self.sess}  # , 'saver': self.saver
        for k, v in kw_self.items():
            if k not in kwargs:
                kwargs[k] = v
        return restore(**kwargs)

    def close(self):
        self.sess.close()
        tf.reset_default_graph()