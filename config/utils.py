import os, re, sys, glob, types, itertools
import numpy as np
from collections import defaultdict

def _xor(a, b):
    return not ((a and b) or (not a and not b))
def _is_property(obj, name):
    return isinstance(getattr(type(obj), name, None), property)
def _is_method(x):
    return type(x) in [types.MethodType, types.FunctionType]
def _is_float(x):
    try:
        float(x)
        return True
    except:
        pass
    return False
def _raise(ex):
    raise ex

_data_alias = {
    'sensat': 'sensaturban',
}
def _import_module(name, cfg_dir='config'):  # import config.name
    import importlib
    name = _data_alias[name] if name in _data_alias else name
    try:
        mod = importlib.import_module(f'config.{name}')
    except Exception as e:
        msg = ''
        if os.path.isdir(cfg_dir) and f'{name}.py' not in os.listdir(cfg_dir):
            cfgs = [i for i in os.listdir(cfg_dir) if i.endswith(".py") and "__" not in i]
            msg = f'no {name} in config mod:\t- cfg dir ({cfg_dir}) has {cfgs}'
        raise type(e)(e.message + msg)
    return mod

def gen_config(cfg_gen, store_dict=None, sep='-'):
    """
    add instance of config generator class to config file 
    Args:
        cfg_gen     : py class  - the generator class, whose `_attr_dict` contains config attributes and the possible values
        store_dict  : dict      - the place to put the generated config instance, may use `globals()`
        sep         : str       - seprator in composite attributes, default to `-`
    """
    if isinstance(cfg_gen, (list, tuple)):
        for cg in cfg_gen:
            gen_config(cg, store_dict, sep)
        return

    assert hasattr(cfg_gen, 'idx_name_pre'), f'no idx_name_pre provided in {cfg_gen}'
    attr_dict = cfg_gen._attr_dict.copy()  # not altering the class variable

    for k, v in attr_dict.items():  # scan for composite config
        assert len(v) > 0, f'found empty list of config options: _attr_dict[{k}]'
        if type(v[0]) == list:
            attr_dict[k] = [sep.join([str(i) for i in v_list if str(i)]).strip(sep) for v_list in itertools.product(*v)]

    attr_k = attr_dict.keys()
    attr_v = [attr_dict[k] for k in attr_k]
    attr_i = [list(range(len(v))) for v in attr_v]

    if store_dict is None:
        store_dict = cfg_gen._store_dict = {}
    for idx in itertools.product(*attr_i):
        cfg = cfg_gen(parse=False)
        for i, k, v in zip(idx, attr_k, attr_v):
            setattr(cfg, k, v[i])
        cfg_var_name = cfg.idx_name_pre + '_' + ''.join([str(i) for i in idx])  # use index to attribute value as name postfix
        setattr(cfg, 'idx_name', cfg_var_name)
        cfg.parse()  # static parse after setting attrs
        store_dict[cfg_var_name] = cfg

def is_config(cfg, base=None, mod=None):
    if mod != None and type(cfg) == str:
        if cfg.startswith('_'):
            return False
        cfg = getattr(mod, cfg)
    if base == None:
        assert mod != None, 'must provide either `base` (class Base) or `mod` (python module)'
        base = mod.Base
    return isinstance(cfg, base) or isinstance(type(cfg), base)  # config can be either class or instance

def log_config(config, title='', f_out=None, prefix='', base=None):
    if f_out is None:
        f_out = sys.stdout
    if base is None:
        root = os.path.join(os.getcwd(), os.path.dirname(__file__), '../')
        sys.path += [] if root in sys.path or os.path.realpath(root) in sys.path else [root]
        from config.base import Base as base

    print(f'\n{prefix}<<< ======= {config._cls} ======= {title if title else config.name}', file=f_out)
    max_len = max([len(k) for k in dir(config) if not k.startswith('_')] + [0])
    for k in config.keys():  # dir would sort
        # if k.startswith('_') or _is_method(getattr(config, k)):
        #     continue
        cur_attr = getattr(config, k)
        if isinstance(cur_attr, list) and len(str(cur_attr)) > 200:  # overlong list
            cur_attr = '[' + f'\n{prefix}\t\t'.join([''] + [str(s) for s in cur_attr]) + f'\n{prefix}\t]'

        print('\t%s%s\t= %s' % (prefix + k, ' ' * (max_len-len(k)), str(cur_attr)), file=f_out)
        if is_config(cur_attr, base=base):
            log_config(cur_attr, f_out=f_out, prefix=prefix+'\t', base=base)
    print('\n', file=f_out, flush=True)

def load_config(cfg_path=None, dataset_name=None, cfg_name=None, cfg_group=None, reload=True):
    # cfg from path
    if cfg_path is not None:
        update = None
        if os.path.isfile(cfg_path):
            # update on the default cfg
            from config.base import Base, Config
            update = Base(cfg_path)
            cfg_path = [update.dataset.lower(), 'default']
        else:
            # directly specified cfg
            cfg_path = cfg_path.replace('/', '.').split('.')
        cfg_path = cfg_path if cfg_path[0] == 'config' else ['config'] + cfg_path
        cfg_module = cfg_path[1]
        cfg_class = '.'.join(cfg_path[2:])
        mod = _import_module(cfg_module)
        if hasattr(mod, cfg_class):
            cfg = getattr(mod, cfg_class)
        else:
            cfg = load_config(dataset_name=cfg_path[1], cfg_name=cfg_class, reload=reload)

        if update is not None:
            cfg = Config(cfg)  # avoid overriding
            cfg.update(update, exclude=[])  # full override with no exclude
        return cfg

    # setup dict
    cfg_name_dict   = load_config.cfg_name_dict    # dataset_name -> {cfg.name -> cfg.idx_name}
    cfg_module_dict = load_config.cfg_module_dict  # dataset_name -> cfg_module

    if dataset_name is not None and dataset_name not in cfg_module_dict or reload:
        mod = _import_module(dataset_name)
        cfg_module_dict[dataset_name] = mod
        cfg_name_dict[dataset_name] = {}
        for i in dir(mod):
            if not is_config(i, mod=mod):  # use the 'base' class imported in 'mod'
                continue
            cfg = getattr(mod, i)
            if cfg.name:
                cfg_name_dict[dataset_name][cfg.name] = cfg.idx_name

    # module/cfg from dataset/cfg name
    mod = cfg_module_dict[dataset_name]
    if cfg_name is not None:
        if cfg_name not in cfg_name_dict[dataset_name]:
            raise KeyError(f'no cfg_name = {cfg_name} in module {dataset_name}')
        idx_name = cfg_name_dict[dataset_name][cfg_name]
        return getattr(mod, idx_name)
    elif cfg_group is not None:
        if not hasattr(mod, cfg_group):
            raise KeyError(f'no cfg_group = {cfg_group} in module {dataset_name}')
        cfg_g = getattr(mod, cfg_group)
        if isinstance(cfg_g, type(mod.Base)) and cfg_g._store_dict:
            cfg_g = cfg_g._store_dict
        if not isinstance(cfg_g, (tuple, list, dict, set)):
            raise ValueError(f'cfg_group = {cfg_group} appears to be {cfg_g}, not of type (tuple, list, dict, set)')
        return cfg_g
    return mod
load_config.cfg_module_dict = {}
load_config.cfg_name_dict = {}

def get_snap(saving_path, step='last', snap_prefix='snap'):
    # get the best of running val (done in training)
    snap_path = os.path.join(saving_path, 'snapshots') if not saving_path.endswith('snapshots') else saving_path
    snap_steps = [f.split('.')[0].split('-')[-1] for f in os.listdir(snap_path) if f.startswith(snap_prefix)]
    if step == 'last':
        snap_steps = sorted([int(s) for s in snap_steps if s.isdigit()]) + sorted([s for s in snap_steps if not s.isdigit()])
        chosen_step = snap_steps[-1]  # last saved snap (best val estimation)
        chosen_snap = os.path.join(snap_path, f'snap-{chosen_step}')
    else:
        assert isinstance(step, int) or step.isdigit() or step == 'best', f'not supported step = {step}'
        step = str(step)
        chosen_snap = None
        if step in snap_steps:
            chosen_snap = os.path.join(snap_path, f'snap-{step}')
        else:
            raise ValueError(f'step={step} not in {snap_steps} (path={snap_path})')
    return chosen_snap

def _list_config(FLAGS):

    cfg = FLAGS.list.replace('/', '.').split('.')
    cfg = [i for i in cfg if i != 'config']
    if os.path.isfile(FLAGS.list):
        cfg = load_config(FLAGS.list)
        mod = _import_module(cfg.dataset.lower())
        cfg_list = [(cfg, cfg.name)]
    elif len(cfg) == 1:  # all config in the dataset_name
        mod = _import_module(cfg[0])
        cfg_list = [(getattr(mod, k), k) for k in dir(mod) if not k.startswith('_')]
        cfg_list = [(c, k) for c, k in cfg_list if is_config(c, mod=mod)]
    elif cfg[0] == 'blocks':
        mod = _import_module(cfg[0])
        blk = '.'.join(cfg[1:])
        cfg_list = [(getattr(mod, 'get_block_cfg')(blk), blk)]
    else:  # config.dataset.dict_name/cfg_name/idx_name
        mod = _import_module(cfg[0])
        try:
            d = load_config(FLAGS.list)
        except:  # may specify a dict/class
            if hasattr(mod, cfg[-1]):
                d = getattr(mod, cfg[-1])
            else:
                raise ValueError(f'no cfg - {FLAGS.list}')
            if issubclass(d, mod.Base) and d._store_dict:
                d = d._store_dict
        cfg_list = zip(d.values(), d.keys()) if isinstance(d, dict) else [(d, d.name)]

    for c, n in cfg_list:
        if FLAGS.set:
            for arg in FLAGS.set.replace(',', ';').split(';'):
                c.update(arg)
        log_config(c, n, base=mod.Base)  # config.Base is mod.Base

if __name__ == '__main__':
    import numpy as np
    import pickle, argparse, time, sys, os, re, glob, shutil
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), '../'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, help='print the cfg group (dict_name, or dataset_name.dict_name)')
    parser.add_argument('--result_path', type=str, default='results', help='root dir to search models')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir name - in case of different version/validation_split')
    parser.add_argument('--set', type=str, help='setting the cfg')
    FLAGS = parser.parse_args()

    cfg_dir = os.path.join(os.getcwd(), os.path.dirname(__file__)).rstrip('/')
    sys.path.insert(0, os.path.dirname(cfg_dir))

    dir_list = None
    if FLAGS.list:
        _list_config(FLAGS)
