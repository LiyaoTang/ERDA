import os, re, sys, copy, warnings
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from collections import defaultdict
from config import log_config, load_config, get_block_cfg
from utils.logger import print_dict
from .heads import resnet_classification_head, resnet_scene_segmentation_head, resnet_multi_part_segmentation_head
from .backbone import resnet_backbone
from .blocks import get_block_ops, apply_block_ops
from .head import apply_head_ops
from .utils import tf_scope
from .basic_operators import *

class Model(object):

    def get_inputs(self, inputs):
        config = self.config
        if isinstance(inputs, dict):
            pass
        else:
            flat_inputs = inputs
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]
            inputs = self.inputs
        for k in ['points', 'neighbors', 'pools', 'upsamples']:
            inputs[k] = [i if i is not None and i.shape.as_list()[0] != 0 else None for i in inputs[k]]
        inputs['sample_idx'] = {
            'down': inputs['pools'],
            'up': inputs['upsamples']
        }

        if 'batches_len' in inputs:
            if 'batches_stack' not in inputs:
                inputs['batches_stack'] = [inputs['in_batches']] + [None] * (config.num_layers - 2) + [inputs['out_batches']]
            if 'batches_ind' not in inputs:
                inputs['batches_ind'] = [inputs['in_batch_inds']] + [None] * (config.num_layers - 1)
        if '_glb' not in inputs:
            inputs['_glb'] = {}  # per-model/device global storage
        # inputs['assert_ops'] = []
        return inputs

    def get_result(self):
        # keys=['logits', 'probs', 'labels']
        # head_rst = {h: {k: d[k] for k in keys if k in d} for h, d in self.head_dict['result'].items()}
        head_rst = self.head_dict['result']
        rst = {  # {head/task: {probs, labels}, ..., 'inputs': input related}
            **head_rst,
            'inputs': {
                'point_inds': self.inputs['point_inds'],
                'cloud_inds': self.inputs['cloud_inds'],                
            }
        }
        for k in ['batches_len']:
            if k in self.inputs:
                rst['inputs'][k] = self.inputs[k]
        return rst

    def get_loss(self):
        return self.loss_dict

    """
    TODO: to check - multiple keys indexing the inputs['point_labels'] should be having the same id in rst - ensure only one tensor passed from gpu to cpu <=
    """

    @tf_scope
    def build_backbone(self, features, block_list, verbose=True):

        # building backbone blocks
        inputs = self.inputs
        config = self.config
        num_layers = config.num_layers
        def is_new_stage(blk):
            if any([k in blk for k in ['pool', 'strided']]):
                return 'down'
            elif any([k in blk for k in ['upsample']]):
                return 'up'
            else:
                return ''

        if 'stage_list' not in inputs:
            down_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(num_layers)]
            up_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(num_layers)] if num_layers > 0 else down_list
            stage_list = {'down': down_list, 'up': up_list}
        else:
            stage_list = inputs['stage_list']
            down_list, up_list = stage_list['down'], stage_list['up']
        inputs['stage_list'] = stage_list

        # backbone - init setting
        stage_i = 0
        block_i = 0
        stage_sc = 'down'
        F_list = down_list
        F_list[stage_i]['p_sample'] = inputs['points'][stage_i]
        F_list[stage_i]['f_sample'] = features
        d_out = config.architecture_dims[0]

        if verbose:
            print(f'\n\n==== {stage_sc}_{stage_i} - arch main')
        for block_cfg in block_list:
            
            block_n = block_cfg.name
            stage_n = is_new_stage(block_n)

            # change stage - indexing the stage after down/up-sampling ops
            if stage_n:

                if verbose:
                    print('---- pts & features')
                    print_dict(F_list[stage_i], prefix='\t')

                # update
                if stage_n == 'down':
                    stage_i += 1

                elif stage_n == 'up':
                    stage_i -= 1

                else:
                    raise NotImplementedError(f'non supported stage name {stage_n}')

                # prepare
                block_i = 0
                stage_sc = stage_n
                F_list = stage_list[stage_n]
                d_out = config.architecture_dims[stage_i]
                kr = config.kr_search[stage_i]
                self.prepare_points(stage_n, stage_i, inputs, config, name=f'{stage_sc}_{stage_i}')

                if verbose:
                    print(f'\n\n==== {stage_sc}_{stage_i} - arch main')
                    print_dict({k: v[stage_i] for k, v in inputs.items() if isinstance(v, tuple)}, prefix='\t')
                    print(f'\td_out = {d_out}; kr = {kr}\n')

            if verbose:
                log_config(block_cfg)

            # special block
            if block_n.startswith('__') and block_n.endswith('__'):
                if block_n == '__up__':
                    block_i = 0
                    stage_sc = 'up'
                    F_list = up_list
                    F_list[stage_i]['p_sample'] = inputs['points'][stage_i]
                    F_list[stage_i]['f_sample'] = features
                else:
                    raise ValueError(f'not supported special block {block_n}')

            # block ops
            else:
                with tf.variable_scope(f'{stage_sc}_{stage_i}/{block_n}_{block_i}'):
                    block_ops = get_block_ops(block_n)
                    features = block_ops(features, d_out, inputs, stage_n, stage_i, block_cfg, config, self.is_training)
                block_i += 1

                if verbose:
                    print(f'{block_n}_{block_i}\t{features}')

            # save the sampled pt/feature (1st block to sample the p_in/f_in of a stage)
            # NOTE update of inputs done in the ops - e.g. changing pt dyanmically based on feature & spatial sampling in inputs
            if stage_n:
                F_list[stage_i]['p_sample'] = inputs['points'][stage_i]
                F_list[stage_i]['f_sample'] = features
            # save as last block
            F_list[stage_i]['p_out'] = inputs['points'][stage_i]
            F_list[stage_i]['f_out'] = features

        # align most downsampled stage in up-down?
        if all(v == None for k, v in up_list[-1].items()):
            up_list[-1] = down_list[-1]
        if verbose:
            print('---- pts & features')
            print_dict(F_list[stage_i], prefix='\t')
            print_dict({'\nstage list =': stage_list})
        return stage_list

    @tf_scope
    def prepare_points(self, stage_n, stage_i, inputs, config):
        # fixed sampling & searching on points - preparing inputs for next stage
        # (may otherwise be specified as block)
        stage_list = inputs['stage_list']
        assert stage_n in ['up', 'down', ''], f'should not invoke prepare_points with stage_n=\'{stage_n}\''
        from ops import TF_OPS

        # if config.debug:
        #     print_dict(inputs, head=f'{stage_n}-{stage_i}')
        #     print(stage_n == 'down' and inputs['points'][stage_i] is None and config.sample in TF_OPS.fix_sample)
        #     print(stage_n == 'down' and inputs['neighbors'][stage_i] is None and config.search in TF_OPS.fix_search)
        #     print(stage_n == 'down' and inputs['sample_idx']['down'][stage_i] is None and config.search in TF_OPS.fix_search)
        #     print(stage_n == 'up' and inputs['sample_idx']['up'][stage_i] is None and config.search in TF_OPS.fix_search)

        # downsampling
        if stage_n == 'down' and inputs['points'][stage_i] is None and config.sample in TF_OPS.fix_sample:
            stage_last = stage_i - 1  # last downsampled stage
            # stage_last = len([i for i in inputs['points'] if i is not None])
            points = stage_list['down'][stage_last]['p_out']
            batches_len = inputs['batches_len'][stage_last] if 'batches_len' in inputs else None
            r = config.r_sample[stage_last]
            rst = TF_OPS.tf_fix_sample(points, r, config.sample, batches_len, verbose=False, name=config.sample)
            if 'batches_len' in inputs:
                inputs['points'][stage_i], inputs['batches_len'][stage_i] = rst
            else:
                inputs['points'][stage_i] = rst

        # neighborhood search
        if inputs['neighbors'][stage_i] is None and config.search in TF_OPS.fix_search:
            points = inputs['points'][stage_i]  # current stage
            batches_len = inputs['batches_len'][stage_i] if 'batches_len' in inputs else None
            kr = config.kr_search[stage_i]
            inputs['neighbors'][stage_i] = TF_OPS.tf_fix_search(points, points, kr, config.search, batches_len, batches_len, name=config.search)

        # downsampling - pool
        if stage_n == 'down' and inputs['sample_idx']['down'][stage_i - 1] is None and config.search in TF_OPS.fix_search:
            stage_last = stage_i - 1  # last downsampled stage
            queries, supports = inputs['points'][stage_i], stage_list['down'][stage_last]['p_out']
            queries_len = supports_len = None
            if 'batches_len' in inputs:
                queries_len, supports_len = inputs['batches_len'][stage_i], inputs['batches_len'][stage_last]
            kr = config.kr_sample[stage_last]
            inputs['sample_idx']['down'][stage_last] = TF_OPS.tf_fix_search(queries, supports, kr, config.search, queries_len, supports_len, name=f'{config.search}_down')

        # upsampling - unpool
        elif stage_n == 'up' and inputs['sample_idx']['up'][stage_i + 1] is None and config.search in TF_OPS.fix_search:
            stage_last = stage_i + 1 - config.num_layers  # last upsampled stage
            # stage_last = [i for i, stage_d in enumerate(stage_list['up']) if stage_d['p_out'] is not None]
            # stage_last = stage_last[0] if stage_last else -1
            queries = stage_list['down'][stage_i]['p_out']
            supports = stage_list['up'][stage_last]['p_out']
            supports = supports if supports is not None else stage_list['down'][-1]['p_out']  # or, the most downsampled
            queries_len = supports_len = None
            if 'batches_len' in inputs:
                queries_len, supports_len = inputs['batches_len'][stage_i], inputs['batches_len'][stage_last]
            kr = config.kr_sample_up[stage_last]
            inputs['sample_idx']['up'][stage_last] = TF_OPS.tf_fix_search(queries, supports, kr, config.search, queries_len, supports_len, name=f'{config.search}_up')

        # if self.config.debug:
        #     print_dict(inputs, head=f'{stage_n}-{stage_i} - prepared', except_k='stage_list')
        #     print('-' * 60)

        return

    @tf_scope
    def build_head(self, head_list, verbose=True):

        # building ouput heads & losses
        head_dict = self.inputs['head_dict'] if 'head_dict' in self.inputs else {'loss': {}, 'result': {}, 'config': {}}
        head_list = head_list if isinstance(head_list, (tuple, list)) else [head_list]
        head_list = [load_config(dataset_name='head', cfg_name=h) if isinstance(h, str) else h for h in head_list]

        if verbose:
            print('\n\n==== arch output')
        for head_cfg in head_list:
            if verbose:
                log_config(head_cfg)
                # if self.config.debug:
                #     print_dict(self.inputs)
            with tf.variable_scope(f'output/{head_cfg.head_n}'):
                head_rst = apply_head_ops(self.inputs, head_cfg, self.config, self.is_training)
            if verbose:
                print_dict(head_rst)

            # loss
            head_k = head_cfg.task if head_cfg.task else head_cfg.head_n  # head for specified task, or head_n as key by default
            loss_keys = ['loss',]
            for k in loss_keys:
                head_rst_d = head_rst[k] if isinstance(head_rst[k], dict) else {head_k: head_rst[k]}  # use returned dict if provided
                joint = head_dict[k].keys() & head_rst_d.keys()
                assert len(joint) == 0, f'head rst {k} has overlapping keys {joint}'
                head_dict[k].update(head_rst_d)
            # result
            rst_keys = ['logits', 'probs', 'labels',]
            head_rst_d = {k: head_rst[k] for k in head_rst if k not in loss_keys}
            assert head_cfg.head_n not in head_dict['result'], f'duplicate head {head_cfg.head_n} in dict'
            assert set(head_rst_d.keys()).issuperset(set(rst_keys)), f'must include keys {rst_keys}, but given {head_rst_d.keys()}'
            head_dict['result'][head_cfg.head_n] = head_rst_d
            if head_k and head_k != head_cfg.head_n:  # get the task head - flat & overridable
                if head_k in head_dict['result']:
                    warnings.warn(f'duplicate task head {head_k} in dict, override by {head_cfg.head_n}')
                head_dict['result'][head_k] = {k: head_rst_d[k][head_k] if isinstance(head_rst_d[k], dict) else head_rst_d[k] for k in head_rst_d}
            # config
            head_dict['config'][head_cfg.head_n] = head_cfg
            head_dict['config'][head_k] = head_cfg

        if verbose:
            print('\n\n')
        return head_dict

    @tf_scope
    def build_loss(self, scope=None, head_dict=None):
        # finalizing loss_dict
        if head_dict is None:
            head_dict = self.head_dict
        loss_dict = head_dict['loss']
        sum_fn = tf.accumulate_n if len(self.config.gpu_devices) else tf.add_n  # accumulate_n seems not working with cpu-only

        # get the collection, filtering by 'scope'
        l2_loss = tf.get_collection('weight_losses', scope)
        if l2_loss and self.config.optimizer not in ['adamW']:
            loss_dict['l2_loss'] = sum_fn(l2_loss, name='l2_loss')  # L2

        # sum total loss
        loss = sum_fn(list(loss_dict.values()), name='loss')

        # reconstruct loss dict - reorder & incldue total loss
        main_n = {'seg': ['S3DIS', 'ScanNet', 'Semantic3D', 'NPM3D', 'ShapeNet', 'PartNet', 'SensatUrban', 'SemanticKITTI']}
        main_n = {v: k for k, lst in main_n.items() for v in lst}[self.config.dataset]
        loss_dict = {
            'loss': loss,
            # # should have one and only one 'main' loss
            # # TODO: may introduce cls & seg head at the same time? => each task a main?
            # main_n: loss_dict.pop(main_n),
            **loss_dict,
        }
        head_dict['loss'] = loss_dict
        return loss_dict

class SceneSegModel(Model):
    def __init__(self, flat_inputs, is_training, config, scope=None, verbose=True):
        self.config = config
        self.is_training = is_training
        self.scope = scope
        self.verbose = verbose

        with tf.variable_scope('inputs'):
            self.inputs = self.get_inputs(flat_inputs)

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']
            self.down_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
            self.up_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
            self.stage_list = self.inputs['stage_list'] = {'down': self.down_list, 'up': self.up_list}
            self.head_dict = self.inputs['head_dict'] = {'loss': {}, 'result': {}, 'config': {}}

            for i, p in enumerate(self.inputs['points']):  # fill points
                self.down_list[i]['p_out'] = p
                # up 0 = the most upsampled, num_layers-1 the upsampled pt from the most downsampled
                self.up_list[i]['p_out'] = p if i < self.num_layers - 1 else None

        if config.dense_by_conv:
            dense_layer.config = config

        with tf.variable_scope('model'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)

            F_up, head = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
                                                        is_training=is_training, init=config.init,
                                                        weight_decay=config.weight_decay,
                                                        activation_fn=config.activation_fn,
                                                        bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

            for i, p in enumerate(self.inputs['points']):  # fill features
                self.down_list[i]['f_out'] = F[i]
                # F_up reversed - 0 = the most upsampled, num_layers-1 the upsampled pt from the most downsampled
                self.up_list[i]['f_out'] = F_up[i] if i < len(F_up) else None
            self.up_list[-1] = self.down_list[-1]  # align the most-downsampled layer
            if head is not None:
                latent, logits = head
                self.up_list[0]['latent'] = latent
                self.up_list[0]['logits'] = logits

            self.head_dict = self.build_head(self.config.arch_out, verbose=verbose)
            self.loss_dict = self.build_loss(scope)
        return

class ModelBuilder(Model):
    def __init__(self, flat_inputs, is_training, config, scope=None, verbose=True):
        self.config = config
        self.is_training = is_training
        self.scope = scope  # variable scope - potential sharing across devices (e.g. gpus)
        self.verbose = verbose

        with tf.variable_scope('inputs'):
            self.inputs = self.get_inputs(flat_inputs)

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']

        with tf.variable_scope('model'):
            self.head_dict = self.build_model_plain_split()
            self.loss_dict = self.build_loss(scope=scope)
        return

    def build_model_plain_split(self):
        """
        detect down-/up-sample via ops
        => architecture = [ops, ...]
        """
        config = self.config

        self.down_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
        self.up_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)] if self.num_layers > 0 else self.down_list
        self.stage_list = {'down': self.down_list, 'up': self.up_list}
        self.head_dict = {'loss': {}, 'result': {}, 'config': {}}

        inputs = self.inputs
        inputs['stage_list'] = self.stage_list
        inputs['head_dict'] = self.head_dict

        # split arch: input -> main -> output
        if '__input__' in config.architecture and '__output__' in config.architecture:
            arch_in = config.architecture[:config.architecture.index('__input__')]
            arch_main = config.architecture[len(arch_in) + 1:config.architecture.index('__output__')]
            arch_out = config.architecture[config.architecture.index('__output__') + 1:]
        else:
            arch_in = config.arch_in
            arch_main = config.arch_main
            arch_out = config.arch_out
        assert len(arch_in) and len(arch_out), f'invalid split of architecture {config.architecture}'
        arch_in = [get_block_cfg(blk) if isinstance(blk, str) else blk for blk in arch_in]
        arch_main = [get_block_cfg(blk) if isinstance(blk, str) else blk for blk in arch_main]
        arch_out = [load_config(dataset_name='head', cfg_name=a) for a in arch_out]

        # arch input
        features = inputs['features']
        self.prepare_points('', 0, inputs, config)
        arch_in_dims = config.arch_in_dims if config.arch_in_dims else [config.first_features_dim] * len(arch_in)
        if self.verbose:
            print(f'\n\n==== inputs')
            print_dict(inputs, prefix='\t', except_k=['stage_list'])
            print('\n\n==== arch input')
        for block_i, (block_cfg, d_out) in enumerate(zip(arch_in, arch_in_dims)):
            with tf.variable_scope(f'input/{block_cfg.name}_{block_i}'):
                features = apply_block_ops(features, d_out, inputs, '', 0, block_cfg, config, self.is_training)
            if self.verbose:
                print(f'{block_cfg.name}_{block_i}\t{features}')

        # arch main - blocks
        self.stage_list = self.build_backbone(features, arch_main, verbose=self.verbose)

        # arch output - pred heads & losses
        self.head_dict = self.build_head(arch_out, verbose=self.verbose)

        return self.head_dict

    def build_model_plain(self):
        """__cfg__ to denote cfg stage
        """
        config = self.config

        self.down_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
        self.up_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
        self.stage_list = {'down': self.down_list, 'up': self.up_list}

        inputs = self.inputs
        inputs['stage_list'] = self.stage_list

        skip_stage = ['__input__']
        sampling_stage = ['down', 'up']
        def is_new_stage(blk):
            if any([k in blk for k in ['pool', 'strided']]):
                return 'down'
            elif any([k in blk for k in ['upsample']]):
                return 'up'
            elif blk in skip_stage:
                return blk
            else:
                return ''

        # init setting
        features = inputs['features']
        # -- stage
        stage_i = 0  # indexing stage cfg (r, inputs)
        block_i = 0  # indexing blocks inside each stage
        stage_sc = 'down'  # scope of the down-/up-sampling stage
        F_list = self.down_list  # collecting feature at each stage
        r = config.first_subsampling_dl * config.density_parameter  # first neighbors search - radius/knn
        # -- arch
        cfg_i = 0  # indexing arch cfg (d_out)
        d_out = config.first_features_dim  # first dim

        if self.verbose:
            print(f'\n==== inputs')
            print_dict(inputs, prefix='\t')
            print(f'\td_out = {d_out}; r = {r}\nfeatures =\t{features}')

        # build blocks & stages
        for block_cfg in config.architecture_cfg[:-1]:

            block_n = block_cfg.name
            stage_n = is_new_stage(block_n)

            # change stage
            if stage_n:

                # update sampling stage
                if stage_n in sampling_stage:

                    # save
                    F_list[stage_i]['p_out'] = inputs['points'][stage_i]
                    F_list[stage_i]['f_out'] = features
                    if self.verbose:
                        print('---- pts & features')
                        print_dict(F_list[stage_i], prefix='\t')

                    # prepare - stage
                    block_i = 0
                    stage_sc = stage_n
                    F_list = self.stage_list[stage_n]
                    if stage_n == 'down':
                        stage_i += 1
                        r *= 2
                    elif stage_n == 'up':
                        stage_i -= 1
                        r /= 2
                    else:
                        raise NotImplementedError(f'invalid sampling stage {stage_n}')

                # special stage, no actual ops
                elif stage_n in skip_stage:
                    pass

                else:
                    raise NotImplementedError(f'non supported stage name {stage_n}')

                # prepare - shared cfg
                cfg_i += 1 if stage_sc == 'down' else -1
                d_out = config.architecture_dims[cfg_i]

            if self.verbose:
                if stage_n in sampling_stage:  # start of new down-/up-sampling stage
                    print(f'\n==== {stage_n}_{stage_i}')
                    print_dict({k: v[stage_i] for k, v in inputs.items() if isinstance(v, tuple)}, prefix='\t')
                    print(f'\td_out = {d_out}; r = {r}')
                elif stage_n in skip_stage:  # skip stage
                    print(f'\n==== {stage_n}')
                    print(f'\td_out = {d_out}; r = {r}')

            # special stage - skip as no ops
            if stage_n in skip_stage:
                continue

            # block ops
            with tf.variable_scope(f'{stage_sc}_{stage_i}/{block_n}_{block_i}'):
                block_ops = get_block_ops(block_n)
                features = block_ops(features, d_out, inputs, stage_n, stage_i, block_cfg, config, self.is_training)
            block_i += 1

            if self.verbose:
                print(f'{block_n}_{block_i}\t{features}')

            # save the sampled pt/feature (1st block to sample the p_in/f_in of a stage)
            # NOTE update of inputs done in the ops - e.g. changing pt dyanmically based on feature & spatial sampling in inputs
            if stage_n:
                F_list[stage_i]['p_sample'] = inputs['points'][stage_i]
                F_list[stage_i]['f_sample'] = features

        # save the last stage
        F_list[stage_i]['p_out'] = inputs['points'][stage_i]
        F_list[stage_i]['f_out'] = features
        if self.verbose:
            print('---- pts & features')
            print_dict(F_list[stage_i], prefix='\t')

        # last block forced to generate output (logits)
        block_cfg = config.architecture_cfg[-1]
        block_n = block_cfg.name
        with tf.variable_scope(f'{stage_sc}_{stage_i}/{block_n}_{block_i}'):
            features = block_ops(features, config.num_classes, inputs, stage_n, stage_i, block_cfg, config, self.is_training)

        if self.verbose:
            print(f'{block_n}_{block_i}\t{features}')
            print_dict({'stage list =': self.stage_list})

        return features


def get_model(flat_inputs, is_training, config, scope=None, verbose=True):
    model = get_inference_model(flat_inputs, is_training, config, scope=scope, verbose=verbose)
    return model

def get_inference_model(flat_inputs, is_training, config, scope=None, verbose=True):
    if config.builder:
        Model = globals()[config.builder]
    elif config.dataset in ['S3DIS', 'ScanNet', 'SensatUrban']:
        Model = ModelBuilder
    else:
        raise NotImplementedError(f'not supported model for dataset {config.dataset}')
    return Model(flat_inputs, is_training, config, scope=scope, verbose=verbose)
