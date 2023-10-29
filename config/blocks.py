"""
config for blocks
"""
# import os, sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path = ([] if ROOT_DIR in sys.path else [ROOT_DIR]) + sys.path

import re
from .base import Base, ConfigList
from .utils import log_config

class Block(Base):
    # import yaml
    # blocks_dict = yaml.load('blocks.yaml', Loader=yaml.FullLoader)
    # def __init__(self, block_n):
    #     # get inheritence list
    #     base_list = [self.blocks_dict[block_n]]
    #     while '_base' in cfg:
    #         b = self.blocks_dict[cfg['_base']]
    #         base_list.append(b)
    #     # update from base
    #     for b in base_list[::-1]:
    #         self.update(b)
    _cls = 'Block'
    def parse(self, attr):
        for a in attr.split('_'):
            k, v = a.split('-')[0], '-'.join(a.split('-')[1:])
            if k:
                setattr(self, k, v)
    @property
    def name(self): return self._name if self._name else self.__class__.__name__
    @name.setter
    def name(self, v): self._name = v

class no_parse(Block):
    def parse(self, attr):
        assert not attr
class __cfg__(no_parse):
    pass

class unary(Block):
    @property
    def linear(self):
        return self.name == 'linear'
linear = unary

class agg(Block):
    def __init__(self):
        self.agg = 'fj'
        self.reduce = 'sum'
    def parse(self, attr):
        for a in attr.split('-'):
            if a in ['sum', 'mean', 'max']:
                self.reduce = a
            elif a in ['fj']:
                self.agg = a
            else:
                raise NotImplementedError(f'agg - not realized attr with only - connected: {a}')
class pool(agg):
    pass

class distconv(agg):
    def __init__(self):
        super(distconv, self).__init__()
        self.enc = 'd'
        self.norm = 'norm'
        self.reduce = 'dist'
        # self.k = ''
    def parse(self, attr):
        if not attr: return
        for a in attr.split('-'):
            if a in ['dn', 'dp', 'dpn']:
                self.enc = a
            elif a in ['softmax']:
                self.norm = a
            elif a.isdigit():
                self.k = int(a)
            else:
                raise NotImplementedError(f'distconv - not realized attr with only - connected: {a}')
# dist = distconv

class conv(Block):
    def __init__(self):

        self.enc = 'dpn'  # encoding to generate kernel, preceding mlp/linear (if any) for position encoding - e.g. mlp-dpn-df to have [mlp(dpn), df]
        self.kernel = 'linear'  # kernel generation from enc

        # self.encact = None  # default to architecture act

        # kernel generation mlp config
        self.kernelact = ''  # default to architecture act
        self.kernelbn = False
        self.kernelbnl = False  # linear bn
        self.kerneldim = 32  # d_mid for self.kernel, e.g. using 'mlp2'
        self.init = 'fanin'  # kernel initialization
        self.wd = 0  # weight decay on kernel

        self.mh = '1'  # shared channel
        self.norm = ''  # norm on kernel
        self.agg = 'fj' # value to apply kernel
        self.reduce = 'sum'  # reduction type

        self.proj = 'bn-act'   # after convolution - potential mlp/linear
        self.bn = 'bn'
        self.act = ''  # if using different activation then the architecture

    @property
    def kernel_cfg(self):
        cfg = {
            'act': self.kernelact,
            'bn': self.kernelbn,
            'wd': self.wd,
            'init': self.init,
            'linearbn': self.kernelbnl,
        }
        return Base(cfg)

    def parse(self, attr):
        for a in attr.split('_'):
            v = [i for i in a.split('-') if i]
            while len(v) > 0:
                k = v[0]
                if k in ['sum', 'mean', 'max']:
                    self.reduce = k
                elif k in ['softmax', 'norm']:
                    self.norm = k
                elif k.isdigit():
                    self.mh = k
                elif 'mlp' in k or 'linear' in k:
                    self.kernel = k
                elif k and v:
                    setattr(self, k, v)
                else:
                    raise NotImplementedError(f'conv - not realized attr with only - connected: {a}')
                v = v[1:]

class lfa(conv):  # lfa - randlanet
    def __init__(self):
        super(lfa, self).__init__()
        self.enc = 'dp-d-p-fj'
        self.encproj = 'mlp'  # projection on pos enc

        self.init = ''
        self.wd = 0

        self.norm = 'softmax'
        self.agg = 'kernel'

        self.proj = 'mlp'
        self.expand = 2
        self.repeat = 2

    def parse(self, attr):
        for a in attr.split('_'):
            k, v = a.split('-')[0], '-'.join(a.split('-')[1:])
            if k and v:
                setattr(self, k, v)
            elif re.match('e\d+', k):
                self.expand = int(k[1:])
            elif re.match('r\d+', k):
                self.repeat = int(k[1:])
            elif k in ['sum', 'mean', 'max']:
                self.reduce = k
            elif k in ['softmax', 'norm']:
                self.norm = k
            elif k.isdigit():
                self.mh = k
            elif 'mlp' in k or 'linear' in k:
                self.kernel = k
            else:
                raise NotImplementedError(f'lfa - not realized attr with only - connected: {a}')


class sample(Block):
    def __init__(self):
        self.sample = None
    def parse(self, attr):
        self.sample = attr
    @property
    def ops(self):
        attr = self.sample.split('-')
        blk_cls = attr[0]
        attr = '-'.join(attr[1:])

        if blk_cls in ['nst']: return
        elif blk_cls in ['max', 'mean', 'sum']: blk_cls = 'agg'; attr = self.sample
        elif blk_cls in ['dist']: blk_cls = 'distconv'

        blk = globals()[blk_cls]()
        blk.name = blk_cls
        blk.parse(attr)
        return blk

class upsample(sample):
    def __init__(self):
        self.f = None  # transform on features
        self.s = None  # transform on skip connection
        self.sample = None
        self.join = None
        self.squeeze = None

    def parse(self, attr):
        attr = attr.split('_')
        self.sample = attr[0]  # support only ops with all '-' connected attrs
        for a in attr[1:]:
            k, v = a.split('-')[0], '-'.join(a.split('-')[1:])
            if any([a.startswith(n) for n in ['sum', 'concat']]):  # e.g. concat-mlp
                self.join = a
            elif k.startswith('sq') or k.isdigit():
                self.squeeze = int(k.replace('sq', ''))
            elif k in ['f', 's'] and v:
                setattr(self, k, v)
            else:
                raise NotImplementedError(f'upsample - not implemented attr = {a}')


class attention_point(Block):  # attention for point transformer
    def __init__(self):
        # way of generating query/key/value
        self.q = 'mlp'
        self.k = 'mlp'
        self.v = 'mlp-pos'

        self.A = 'minus-pos-mlp2'  # way of generating attention map - need to add 'softmax'
        # self.share = ''
        self.mh = 8  # multi-head - #channel per head
        self.pos = 'mlp2-dp'  # position encoding

        self.reduce = 'sum'  # way of collecting V by A
        # self.scale = True

        self.ratio = None
        self.ffn = None
        # self.ffn_ratio = None
        self.act = 'relu'
        self.wd = 0

    def parse(self, attr):
        for v in [a.split('-') for a in attr.split('_')]:
            while len(v) > 0:
                k = v[0]
                if k in ['sum', 'mean', 'max']:
                    self.reduce = k
                elif any([k.startswith(i) for i in ['drop', 'rescale', 'scale', 'softmax', 'norm']]):
                    self.A = self.A + f'-{k}'
                elif re.fullmatch('r\d+', k):
                    self.ratio = int(k[1:])
                elif re.fullmatch('ffn\d+(|r\d+)', k):
                    if 'r' in k:
                        self.ffn_ratio = int(k.split('r')[-1])
                        k = k.split('r')[0]
                    self.ffn = f'mlp{k[3:]}'
                elif len(v) > 1:
                    setattr(self, k, '-'.join(v[1:]))
                    break
                else:
                    raise NotImplementedError(f'attention - not realized attr with only - connected: {a}')
                v = v[1:]

class resnetb(Block):
    def __init__(self):
        self.depth = 1
        self.ratio = 4
        self.stride = 1
        self.ops = None
    def _is_ops(self, attr):
        return any([attr.split('-')[0].startswith(k) for k in ['conv', 'att', 'pool', 'dist', 'lfa']])
    def parse(self, attr):
        attr = attr.split('_')
        ops = [i for i, a in enumerate(attr) if self._is_ops(a)]  # start idx of ops
        for a in attr[:ops[0]]:
            if 'd' in a:
                self.depth = int(a[a.index('d') + 1])
            elif 'r' in a:
                self.ratio = int(a[a.index('r') + 1])
            else:
                raise NotImplementedError(f'resnetb - not supported attr: {a} in {attr}')

        ops = '_'.join(attr[ops[0]:])
        if ';' in ops:
            self.ops = ConfigList([get_block_cfg(o) for o in ops.split(';')])
        else:
            self.ops = get_block_cfg(ops)

class resnetb_strided(resnetb):
    def __init__(self):
        super(resnetb_strided, self).__init__()
        self.stride = 2
        self.ops = None
        self.pool = None
    def _is_ops(self, attr):
        return any([attr.split('-')[0].startswith(k) for k in ['conv', 'lfa', 'att', 'pool', 'max', 'mean', 'dist']])
    def parse(self, attr):
        attr = attr.split('_')  # may split the sub-attr for ops/pool
        ops_pool = [i for i, a in enumerate(attr) if self._is_ops(a)]  # start idx of ops/pool
        for i, a in enumerate(attr[:ops_pool[0]]):
            if a.startswith('r'):
                self.ratio = int(a[1:])
            else:
                raise NotImplementedError(f'resnetb_strided - not supported attr: {a} in {attr}')

        ops = '_'.join(attr[ops_pool[0]:ops_pool[1]])
        if ';' in ops:
            self.ops = ConfigList([get_block_cfg(o) for o in ops.split(';')])
        else:
            self.ops = get_block_cfg(ops)

        pool = '_'.join(attr[ops_pool[1]:])
        self.pool = get_block_cfg(f'sample-{pool}')

def get_block_cfg(block, raise_not_found=True, verbose=False):
    """
    '__xxxx__'  - special block for config use
    '{block_n}-{attr 1}_{attr 2}....': cfg class name - attrs, with multiple attr connected via "_"
    """

    # from . import blocks
    block = block.split('-')
    blk_cls = block[0]
    attr = '-'.join(block[1:])

    if blk_cls.startswith('__') and blk_cls.endswith('__'):
        blk = __cfg__()
    elif blk_cls in globals():
        blk = globals()[blk_cls]()
    elif raise_not_found:
        raise KeyError(f'block not found: {blk_cls} - {attr}')
    else:
        return None
    
    if attr:
        blk.parse(attr)
    if blk._assert:
        blk._assert()

    # # get the default setting
    # blk = Block(blk_cls)
    # # update
    # blk_fn = getattr(blocks, blk_cls)
    # blk = blk_fn(blk, attr)
    if not blk.name:
        blk.name = blk_cls
    if not blk.attr:
        blk.attr = attr
    if verbose:
        log_config(blk)
    return blk