

class dict_list(dict):
    # collect a series of list of dict of same type & append value
    # => list_dict -> dict_list
    def __init__(self, d=None):
        if isinstance(d, (list, tuple)) and len(d) > 0:
            self.init_shape(d[0])
            for dd in d:
                self.append(dd)
        elif isinstance(d, dict):
            self.init_shape(d)
        elif d is not None:
            raise ValueError(f'not supported initilization d = {d}')
    def init_shape(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self[k] = dict_list(v)
            else:
                self[k] = []
    def append(self, d):
        for k, v in d.items():
            self[k].append(v)

class list_dict(list):
    # collect a dict of list of the same len & split value
    # => dict_list -> list_dict
    def __init__(self, ls=None):
        self._max = None
        self._d = None
        if isinstance(ls, dict):
            self.init_shape(ls)
            self.split(ls)
        elif isinstance(ls, (list, tuple)):
            self._max = len(ls)
            self += list(ls)
        elif ls is not None:
            raise ValueError(f'not supported initilization ls = {ls}')
        self.ls = ls

    def init_shape(self, d):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                if self._max is None:
                    self._max = len(v)
                assert len(v) == self._max, ValueError(f'existing len = {self._max}, encountered different len = {len(v)}')
            else:
                assert isinstance(v, dict), ValueError(f'expected dict, but get type {type(v)}, {k}: {v}')
                self.init_shape(v)

    def split(self, d):
        for i in range(self._max):
            self.append(self.get_items(d, i))

    def get_items(self, d, i):
        store = {}
        for k, v in d.items():
            if isinstance(v, dict):
                store[k] = self.get_items(v, i=i)
            else:
                store[k] = v[i]
        return store

def traverse_list(l, fn, inplace=True, try_fn=True, ignore_type=None, ignore_none=False, **kwargs):
    container = l if inplace else [None] * len(l)
    for i, v in enumerate(l):
        if isinstance(v, dict) and not isinstance(v, ignore_type):
            v = traverse_dict(v, fn, inplace, try_fn, ignore_type, **kwargs)
        elif isinstance(v, list):
            v = traverse_list(v, fn, inplace, try_fn, ignore_type, **kwargs)
        elif ignore_none and v is None:
            pass
        elif try_fn:
            try: v = fn(v, **kwargs)
            except: pass
        else:
            v = fn(v, **kwargs)
        container[i] = v
    return container

def traverse_dict(d, fn, inplace=True, try_fn=True, ignore_type=None, ignore_none=False, **kwargs):
    container = d if inplace else {}
    ignore_type = tuple(ignore_type) if ignore_type else ()
    for k, v in d.items():
        if isinstance(v, dict):
            v = traverse_dict(v, fn, inplace, try_fn, ignore_type, **kwargs)
        elif isinstance(v, list) and not isinstance(v, ignore_type):
            v = traverse_list(v, fn, inplace, try_fn, ignore_type, **kwargs)
        elif ignore_none and v is None:
            pass
        elif try_fn:
            try: v = fn(v, **kwargs)
            except: pass
        else:
            v = fn(v, **kwargs)
        container[k] = v
    return container

def dual_traverse_dict(src, dst, fn, inplace=True, try_fn=True, keys=None, **kwargs):
    container = dst if inplace else {}
    keys = keys if keys is not None else src.keys()
    for k in keys:
        v_src = src[k]
        v_dst = dst[k]
        if isinstance(v_src, dict):
            v = dual_traverse_dict(v_src, v_dst, fn, inplace, try_fn, **kwargs)
        elif try_fn:
            try: v = fn(v_src, v_dst, **kwargs)
            except: pass
        else:
            v = fn(v_src, v_dst, **kwargs)
        container[k] = v
    return container

    