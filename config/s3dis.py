import re, itertools
from .base import Base, Config
from .utils import gen_config, _xor, _is_property, _is_float, load_config
from collections import defaultdict

class Default(Config):
    """
    dataset default setting
    """
    dataset = 'S3DIS'

    # ---------------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------------- #
    gpus = 4
    batch_size = 4  # actual running batch (per-gpu) - influence BxN
    batch_size_val = 16
    # epoch setting
    epoch_batch = 8  # batch size per-step - TODO: set to batch_size * gpu_num
    epoch_steps = 500  # desired step per-epoch - #samples (from global point cloud) = epoch_batch (# input point cloud per step) * epoch_steps
    validation_steps = 100  # desired step per validation epoch
    max_epoch = 600
    # loss
    loss_weight = None  # if using weighted loss
    # optimizer
    learning_rate = 0.01  # 0.005 * sqrt(total batch size) / 2
    optimizer = 'sgd'
    momentum = 0.98
    decay_epoch = 1
    decay_rate = 0.9885531
    grad_norm = 100
    grad_raise_none = True
    # saving & io
    num_threads = 12  # thread to provide data
    print_freq = 60
    update_freq = 10
    save_freq = 0 # 50
    val_freq = 10
    save_val = ''  # if saving val results (by keys)
    save_sample = False  # if saving sample stat
    save_compact = True  # if saving only vars
    summary = False  # if using tf summary
    runtime_freq = 0  # runtime stat
    # testing
    num_votes = 20

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    in_radius = 2.0  # sampling data - in_radius = 50 * first_subsampling_dl
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    # augment_scale = .3  # random scaling - 1 [+/- scale]
    augment_scale_min = 0.7
    augment_scale_max = 1.3
    augment_noise = 0.001
    augment_rgb_drop = 0.8
    augment_rgb_trans = ''    # translation on rgb
    augment_rgb_noise = ''    # rand noise (jitter) on rgb
    augment_rgb_contrast = '' # enhance rgb contrast
    augment_rgb_hstrans = ''     # translation on hue-saturation (H, S)

    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    num_classes = 13  # num of valid classes
    in_features_dim = 5  # input point feature type
    in_features = {1: '1', 2: '1-Z', 3: 'rgb', 4: '1-rgb', 5: '1-rgb-Z', 6: '1-rgb-xyz', 7: '1-rgb-xyz-Z'}[in_features_dim]
    first_features_dim = 72
    num_layers = 5
    # sampling & search
    search = 'radius'  # knn/radius search
    sample = 'grid'  # grid/random/fps search
    density_parameter = 5.0
    first_subsampling_dl = 0.04
    # radius - neighbor/pooling (during sub-sampling)/up-sampling
    kr_search = [dl * dp / 2 * (2 ** i) for i, (dl, dp) in enumerate([(first_subsampling_dl, density_parameter)] * num_layers)]
    kr_sample = kr_search[:-1]
    kr_sample_up = [2 * r for r in kr_search[:-1]]  # up-sampling radius
    r_sample = [dl * 2 ** (i+1) for i, dl in enumerate([first_subsampling_dl]*(num_layers-1))]  # ratio/radius of sub-sampling points
    neighborhood_limits = [26, 31, 38, 41, 39, 29]
    # neighborhood_limits = [29, 30, 36, 39, 37, 29]
    # global setting
    activation = 'leaky_relu'  # relu, prelu
    init = 'xavier'
    weight_decay = 0.001
    bn_momentum = 0.99
    bn_eps = 1e-6

    # model architecture: each layer (stage) start from a down-/up-sample ops
    # [
    #   ops='solver-args',
    #   ...
    # ]
    architecture = []
    idx_name = name = 'default'
default = Default()

