import re
import numpy as np

class StepScheduler(object):
    def __init__(self, name, base_value, decay_rate, decay_step, max_steps, clip_min=0):
        self.name = name
        self.clip_min = clip_min
        self.cur_step = 0
        self.values = [base_value * decay_rate ** (i // decay_step) for i in range(max_steps)]

    def reset(self):
        self.cur_step = 0

    def step(self):
        # cur_value = self.base_value * self.decay_rate ** (cur_step // decay_step)
        cur_value = max(self.values[self.cur_step], self.clip_min)
        self.cur_step += 1
        return cur_value

class LrScheduler(object):
    def __init__(self, config):
        self.config = config
        self.start_lr = float(config.learning_rate)
        self.clip_min = config.clip_min if config.clip_min else 0

        self.decay = config.decay
        if self.decay.startswith('cos'):
            self._get_lr = self._get_lr_cos

        self.reset()

        # from matplotlib import pyplot as plt
        # plt.plot(self.to_list(config.max_epoch))
        # plt.savefig(config.name)

    def reset(self):
        self.cur_ep = 0
        self.cur_step = 0
        self.learning_rate = None  # None to denote not initalized
        self.learning_rate = self._get_lr()

    def _get_lr_cos(self):
        # simple implementation for cos annealing (epoch based)
        # borrowing from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
        # e.g. cos_w10, cos_w10_c3_m2_g.5
        cfg = self.config
        cur_ep = self.cur_ep
        total_ep = cfg.max_epoch
        max_lr = self.start_lr
        base_lr = self.clip_min if self.clip_min > 0 else 1e-5  # starting lr (min)

        warm_ep = re.search('w\d+', self.decay)
        warm_ep = float(warm_ep.group()[1:]) if warm_ep else 0
        if 0 < warm_ep and warm_ep < 1:
            warm_ep = total_ep * warm_ep

        # solve cycle
        cycle_ep = re.search('c\d+', self.decay)
        cycle_ep = int(cycle_ep.group()[1:]) if cycle_ep else 0  # total num of cycles
        cycle_m = re.search('m\d+', self.decay)
        cycle_m = float(cycle_m.group()[1:]) if cycle_m else 1  # extending len per cycle
        if cycle_m > 1:
            assert cycle_ep > 0, f'#cycle must > 0'
            cycle_ep_base = total_ep * (cycle_m - 1) / (cycle_m ** cycle_ep - 1)  # solving the first cycle len - sum of geometric sequence (等比求和)
            cycle_ep = [cycle_ep_base * cycle_m ** i for i in range(cycle_ep)]
            cycle_n = len([i for i in np.cumsum(cycle_ep) if i < cur_ep])  # num of cycles
            cycle_base = np.sum(cycle_ep[:cycle_n])  # start ep of current cycle
            cycle_ep = cycle_ep[cycle_n]  # current cycle length
        elif cycle_ep:
            assert total_ep % cycle_ep == 0, f'#cycle={cycle_ep} does not align with #total={total_ep}'
            cycle_ep = total_ep / cycle_ep  # length of each cycle - default to total_ep (1 cycle)
            cycle_n = int(cur_ep / cycle_ep)
            cycle_base = cycle_n * cycle_ep
        else:
            cycle_ep, cycle_n, cycle_base = total_ep, 0, 0
        cur_ep = cur_ep - cycle_base

        # modulate max lr
        gamma = [i[1:] for i in self.decay.split('_') if i.startswith('g')]
        gamma = float(gamma[0]) if gamma else 1
        max_lr = max_lr * gamma ** cycle_n

        if cur_ep < warm_ep:
            # warmup stage - linear increasing
            return cur_ep / warm_ep * (max_lr - base_lr) + base_lr
        else:
            # cos decay stage
            cur_ep = cur_ep - warm_ep
            cycle_ep = cycle_ep - warm_ep
            decay = (1 + np.cos(np.pi * cur_ep / cycle_ep)) / 2  # rescaled cos weight in [0, 1]
            return base_lr + (max_lr - base_lr) * decay

    def _get_lr(self):
        # exponential decay (default)
        cfg = self.config
        cur_ep = self.cur_ep
        base_lr = self.clip_min if self.clip_min > 0 else 1e-5

        warm_ep = re.search('w\d+', self.decay)
        warm_ep = float(warm_ep.group()[1:]) if warm_ep else 0

        if cur_ep < warm_ep:
            # warmup stage - linear increasing
            return cur_ep / warm_ep * (self.start_lr - base_lr) + base_lr

        # normal decay
        cur_ep = cur_ep - warm_ep
        if cfg.decay_step:
            times = self.cur_step // cfg.decay_step if isinstance(cfg.decay_step, int) else (np.array(cfg.decay_step) <= self.cur_step).sum()
        else:
            decay_epoch = cfg.decay_epoch if cfg.decay_epoch else 1  # decay per epoch by default
            if isinstance(decay_epoch, (list, tuple)):
                assert all(i >= 1 for i in decay_epoch), f'need to specify as real epoch, not {decay_epoch}'
            times = cur_ep // decay_epoch if isinstance(decay_epoch, int) else (np.array(decay_epoch) <= cur_ep).sum()

        cum_decay = (cfg.decay_rate ** times) if type(cfg.decay_rate) in [int, float] else np.prod(cfg.decay_rate[:times])  # np.prod([]) = 1.0
        cur_lr = self.start_lr * cum_decay
        return cur_lr

    def to_list(self, max_epoch=None):
        lrs = []
        max_epoch = max_epoch if max_epoch is not None else self.config.max_epoch
        for i in range(max_epoch):
            self.cur_ep = i
            lrs.append(self._get_lr())
            self.learning_rate = lrs[-1]
        self.reset()
        return lrs

    def step(self, epoch, step):
        self.cur_ep += epoch
        self.cur_step += step
        cur_lr = max(self._get_lr(), self.clip_min)
        self.learning_rate = cur_lr
        return cur_lr

    def to_plot(self, max_epoch=None):
        lrs = []
        max_epoch = max_epoch if max_epoch is not None else self.config.max_epoch
        for i in range(max_epoch):
            self.cur_ep = i
            lrs.append(self._get_lr())
            self.learning_rate = lrs[-1]
        self.reset()
        import matplotlib.pyplot as plt
        plt.plot(lrs)
        plt.show()
        return 
