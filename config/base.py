import re, os, sys
from .utils import _is_property, _is_method

def _try_eval(s):
    try:
        s = eval(s)
    except:
        pass
    return s
class _Base(type):
    def __getattr__(self, name):
        return ''

class Base(metaclass=_Base):
    _cls = 'Config'
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        return ''

    # dict-like interface
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, item):
        setattr(self, key, item)
    def __contains__(self, key):
        return hasattr(self, key)

    # # pickle interface
    # def __getstate__(self):
    #     return vars(self)
    # def __setstate__(self, state):
    #     vars(self).update(state)

    def __init__(self, cfg=None, parse=False):
        if cfg:
            self.update(cfg, exclude=[])
        if parse:
            self.parse()

    def keys(self, exclude=[]):  # enable ** operation on Base
        exclude = [exclude] if isinstance(exclude, str) else exclude
        k_list = [k for k in dir(self) if not k.startswith('_') and k not in exclude and not _is_method(getattr(self, k))]
        return k_list

    def dict(self, exclude=[]):
        kv = {}
        exclude = [exclude] if isinstance(exclude, str) else exclude
        for attr in self.keys(exclude=exclude):
            kv[attr] = getattr(self, attr)
        return kv

    def items(self, exclude=[]):
        return list(self.dict(exclude=exclude).items())

    def update(self, cfg, key=None, exclude=['name', 'idx_name']):
        if key != None:  # store cfg as a sub-config
            setattr(self, key, Base(cfg))
        elif isinstance(cfg, str):  # path / dict in str
            if os.path.isfile(cfg) and cfg.endswith('yaml'):
                import yaml
                with open(cfg) as f:
                    cfg = dict(yaml.load(f, Loader=yaml.FullLoader))
            elif cfg.startswith('{') and cfg.endswith('}'):
                cfg = eval(cfg)
            else:
                # cfg = cfg.replace('\:', '：')  # ues chinese char to escape
                # cfg = dict([[i.strip().replce('：', ':') for i in t.split(':')] for t in cfg.split(',')])
                cfg = dict([[i.strip() for i in t.split(':')] for t in cfg.split(',')])
                cfg = {_try_eval(k): _try_eval(v) for k, v in cfg.items()}
            self.update(cfg, exclude=exclude)
        elif isinstance(cfg, dict):  # update from dict
            for k, v in cfg.items():
                if k in exclude:
                    pass
                elif _is_property(self, k) or _is_method(getattr(self, k)):
                    # print(f'{k} is a property / method in {self}', file=sys.stderr)
                    try:
                        setattr(self, k, v)
                    except:
                        raise KeyError(f'{k} is a property / method in {self}')
                elif isinstance(v, dict):  # nesting dict
                    if hasattr(self, k) and isinstance(getattr(self, k), dict):  # replace the dict (if it exists & is indeed a dict)
                        setattr(self, k, v)
                    elif hasattr(self, k) and isinstance(getattr(self, k), Base):  # update the sub-config
                        getattr(self, k).update(v, exclude=exclude)
                    else:  # extend to be a sub-config
                        self.update(v, key=k, exclude=exclude)
                elif isinstance(k, str) and '.' in k:
                    k = k.split('.')
                    kk = k[0]
                    getattr(self, kk).update('.'.join(k[1:]), v)
                else:
                    setattr(self, k, v)
        else:  # update from another cfg
            for attr in [i for i in dir(cfg) if not i.startswith('_') and i not in self._attr_dict]:
                if not _is_property(self, attr) and not _is_method(getattr(cfg, attr)) and attr not in exclude:
                    setattr(self, attr, getattr(cfg, attr))
        return self

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def parse(self):
        pass

    def freeze(self):
        cfg = Base()
        cfg.update(self, exclude=[])  # turn all property into frozen args
        return cfg

    def dump(self, stream=None, exclude=[], **kwargs):
        if isinstance(stream, str):
            if not stream.lower().endswith('.yaml'):
                stream = f'{stream}.yaml'
            stream = open(stream, 'w')
        import yaml
        return yaml.dump(self.dict(exclude=exclude), stream, **kwargs)


class Config(Base):

    # ---------------------------------------------------------------------------- #
    # project-wise setting
    # ---------------------------------------------------------------------------- #
    data_path = 'Data'
    saving_path = None
    saving = True  # save model snap
    save_val = ''  # keys to save
    save_best = True
    save_test = False  # save model inference results after trained
    snap_dir = 'snapshots'
    snap_prefix = 'snap'
    # summary_dir = ''  # use saving_path - assume single train per saving_path
    max_to_keep = 10
    mode = 'train-val'  # -test

    rand_seed = 0
    distribute = 'tf_device'
    colocate_gradients_with_ops = False
    cpu_variables = False

    grad_raise_none = True

    @property
    def gpu_devices(self):
        if isinstance(self.gpus, int) or self.gpus.isdigit():
            cuda_dev = [str(i) for i in range(int(self.gpus))]
        else:
            assert ',' in self.gpus, f'unexpected cfg.gpus = {self.gpus}'
            cuda_dev = [i.strip() for i in self.gpus.split(',') if i.strip()]
        cuda_dev = ','.join(cuda_dev)
        return cuda_dev
    @property
    def gpu_num(self):
        # NOTE: gpu_num=1 when gpus=0, as gpu_devices='' => gpu_devices.split(',')=['']
        # => check if gpu available by gpu_devices == ''
        return len(self.gpu_devices.split(','))
    @property
    def _num_layers(self):
        # Number of layers - downsample_stage, each stage = [downsample ops, ops, ...]
        is_down_stage = lambda blk: any([k in blk for k in ['pool', 'strided']])
        return len([block for block in self.architecture if is_down_stage(block)]) + 1
    @property
    def idx_name_pre(self): return type(self).__name__.lower()

    def __init__(self, cfg=None, parse=True):
        super(Config, self).__init__(cfg, parse)
        if parse:
            self.parse()

    def parse(self):
        if self.architecture:
            assert self.num_layers == self._num_layers, f'claim as {self.num_layers}, but calc to be {self._num_layers}'

        if not self.architecture_dims:
            # d_out - output dims of each stage (first_features_dim = d_in of stage 0)
            self.architecture_dims = [self.first_features_dim * (2 ** i) for i in range(1, self.num_layers + 1)]
        # self.architecture_cfg = super(Config, self).parse()  # parse config

class ConfigList(Base, list):
    _cls = 'ConfigList'
    def __init__(self, iterable, parse=False):
        Base.__init__(self)
        list.__init__(self)

        if parse:
            for c in iterable:
                c.parse()

        for c in iterable:
            self.append(c)
        return

    def __setattr__(self, __name, __value):
        if __name in self.keys():
            raise AttributeError(f'duplicate attr={__name} value={__value}')
        return super().__setattr__(__name, __value)

    def append(self, cfg):
        setattr(self, cfg.name, cfg)  # set before adding cfg for checking
        super().append(cfg)

    def insert(self, i, cfg):
        setattr(self, cfg.name, cfg)
        super().insert(i, cfg)

    def extend(self, cfgs):
        for c in cfgs:
            setattr(self, c.name, c)
        super().extend(cfgs)

    def keys(self, exclude=[]):
        exclude = [exclude] if isinstance(exclude, str) else exclude
        k_list = [c.name for c in self]
        k_list = [k for k in k_list if not k.startswith('_') and k not in exclude and not _is_method(getattr(self, k))]
        return k_list

    def __dir__(self):
        return [c.name for c in self]
