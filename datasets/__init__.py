from .tf_s3dis_dataset import S3DISDataset
from .tf_scannet_dataset import ScanNetDataset
from .tf_sensaturban_dataset import SensatUrbanDataset

def get_dataset(config, *args, **kwargs):
    _cls_n = f'{config.dataset}Dataset'
    return globals()[_cls_n](config, *args, **kwargs)
