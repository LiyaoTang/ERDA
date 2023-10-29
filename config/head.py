"""
config for head - auxilary network + output (label)
"""

import re, itertools
from .base import Base, Config
from .blocks import get_block_cfg
from .utils import gen_config, _xor, _is_property, _is_float, load_config

class Head(Base):
    _cls = 'Head'
    def __init__(self, cfg=None, parse=False):
        self._weight = ''  # loss weight
        self._ftype = 'out'  # pts & features to use (of each stage)
        self._stage = ''  # down/up stage

    # TODO:
    # 1. adding loss with stage & ftype control (no footprint on stage_list)
    # 2. adding loss with label level control (sub-scene loss)
    # 3. potential control on range of sub-scene?

    @property
    def weight(self): return self._weight
    @property
    def ftype(self): return self._ftype
    @property
    def stage(self): return self._stage
    @property
    def head_n(self): return type(self).__name__
    @property
    def idx_name_pre(self): return self.head_n
    @property
    def task(self): return self.head_n
    # @property
    # def _is_main(self): return _is_main_head(self.head_n)

    def parse(self, attr=''):
        if not attr:  # skip static parse - as using gen_config/load_config
            return
        for a in attr.split('_'):
            k, v = a.split('-')[0], '-'.join(a.split('-')[1:])
            if self.common_setter(k):
                pass
            elif k and v:
                setattr(self, k, v)
            else:
                raise NotImplementedError(f'Head Base - not supported a = {a} in attr = {attr}')
    def common_setter(self, a):
        if a in ['sample', 'out']:
            self._ftype = a
        elif any([a.startswith(i)] for i in ['up', 'down', 'U', 'D']):
            self._stage = a
        elif re.fullmatch('w\d+', a):
            self._weight = float(a[1:])
        else:
            return False
        return True

class MLP(Head):
    # cross-entropy loss, by default, on the last (most upsampled) layer
    _attr_dict = {'_ops': [
        '1|xen||', '|||',       # xen - per-pixel softmax with cross entropy,
        '1|xen||class', '|||class',
        '1|xen|dp.5|', '||dp.5|',
        '1|xen|dp.5|class', '||dp.5|class',
    ]}
    act = 'relu'
    task = 'seg'
    ftype = 'f_out'
    stage = ''
    @property
    def mlp(self): t = self._ops.split('|')[0]; return int(t) if t else 1
    @property
    def loss(self): l = self._ops.split('|')[1]; return l if l else 'xen'
    @property
    def weight(self): w = self._ops.split('|')[3]; return w
    @property
    def _reg(self): return self._ops.split('|')[2].split('-')
    @property
    def drop(self):
        dp = [i for i in self._reg if i.startswith('dp')]
        return float(dp[0][2:]) if dp else ''
    @property
    def smooth(self):
        t = [i for i in self._reg if i.startswith('s')]
        return float(t[0][1:]) if t else ''
    @property
    def temperature(self):
        t = [i for i in self._reg if i.startswith('T')]
        return t[0][1:] if t else ''
    @property
    def head_n(self): return type(self).__name__.lower()
    @property
    def name(self): return '-'.join([self.head_n] + [i for i in self._ops.split('|') if i])

    def parse(self, attr=''):
        for a in attr.split('-'):
            if a.isdigit(): self.mlp = int(a)
            elif a in ['xen', 'sigmoid', 'none']: self.loss = a
            elif a != 'pred' and self.common_setter(a): pass
            else: raise NotImplementedError(f'Head mlp: not supported a = {a} in attr = {attr}')

mlp_dict = {}
gen_config([MLP], mlp_dict)
for k, v in mlp_dict.items():
    globals()[k] = v

class pseudo(Head):
    _attr_dict = {'_ops': [
        'fout-pmlp2|mom|normdot|w.1',
        'fout-pmlp2|mom-Gavg|normdot|w.1',
        'fout-pmlp2|mom-Gsum|normdot|w.1',

        'fout-pmlp2|mom-I|normdot|w.1',
        'fout-pmlp2|mom-I|normdot|w.01',
        'fout-pmlp2|mom-I-Gsum|normdot|w.1',
        'fout-pmlp2|mom-I-Gsum|normdot|w.01',

    ]}
    @property
    def _glb(self): return self._ops.split('|')
    # feature - 
    @property
    def _feat(self): return self._glb[0].split('-')
    @property
    def ftype(self): f = [i for i in self._feat if i.islower()]; return f[0]
    @property
    def _proj(self): proj = [f for f in self._feat if f.startswith('p')]; return proj if proj else ''
    @property
    def project(self): return [p[1:] for p in self._proj]
    @property
    def project_fdim(self): proj = [f for f in self._feat if f.startswith('fd')]; return int(proj[0][2:]) if proj else ''
    @property
    def coadapt(self): return True  # True to allow grad on pseudo label - can be derived to be grad@i l_i - p_i * L
    # target - 
    @property
    def _target(self): return self._glb[1].split('-')
    @property
    def momentum_update(self): m = [t for t in self._target if t.lower().startswith('mom')]; n = len(re.search('[a-zA-Z]+', m[0]).group()) if m else 0; return m[0][n:] if m and m[0][n:] else '.999' if m else False
    @property
    def _momentum_update_stage(self):
        m = [t for t in self._target if any(t.startswith(i) for i in ['Gavg', 'Gsum'])]
        if m: return m[0].replace('G', 'glb_')
        return ''
    @property
    def momentum_update_stage(self): return self._momentum_update_stage
    @property
    def momentum_init(self): m = [t for t in self._target if t.startswith('I')]; return m[0][1:] if m else 'zeros'
    @property
    def target_reduce(self): r = [t for t in self._target if any(t.startswith(k) for k in ['sum'])]; return r[0] if r else 'mean'
    # dist - 
    @property
    def _dist(self): return self._glb[2].split('-')
    @property
    def dist(self): return self._dist[0]  # distance
    @property
    def scale(self): s = [i for i in self._dist[1:] if i in ['inv', 'exp', 'negexp', 'neg']]; return s[0] if s else 'neg'
    @property
    def norm(self): return 'softmax'
    @property
    def _loss(self): return self._glb[3].split('-')
    @property
    def loss(self):
        l = [i for i in self._loss if any(i.startswith(ii) for ii in ['kl', 'klr', 'js', 'mse'])]
        return '-'.join(l) if l else 'xen'
    @property
    def weight(self):  return self._loss[-1][1:]

    @property
    def name(self): return f'{self.head_n}-' + '-'.join([i.strip('-') for i in self._glb if i])

weak_dict = {}
gen_config([pseudo], weak_dict)
for k, v in weak_dict.items():
    globals()[k] = v

def get_head_cfg(head):
    """
    NOTE: block_cfg compatible API - not used in architecture building, but using 'load_config'
        => get head cfg by name, instead of dynamically setting attr
        => can have more self-defined group of attrs (e.g. sep = '|'), no need to worry about '-'/'_'

    '{head_n}-{attr 1}_{attr 2}....': cfg class name - attrs, with multiple attr connected via '_'
    """

    head = head.split('-')
    head_cls = head[0]
    attr = '-'.join(head[1:])

    head = globals()[head_cls]()
    if attr:
        head.parse(attr)
    if head._assert:
        head._assert()

    head.name = head_cls
    head.attr = attr
    return head

del k, v

