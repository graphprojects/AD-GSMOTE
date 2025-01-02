import os.path as osp
import sys

import yaml
from easydict import EasyDict

sys.dont_write_bytecode = True


def load_dataset_params(dataset, train_ratio, config_dir):
    config_path = osp.join(
        config_dir,
        "config",
        "dataset_params.yaml",
    )

    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    datname = f"{dataset}_{train_ratio}"
    if datname not in params:
        raise ValueError(f"Dataset {datname} not found in config file")

    return EasyDict(params[datname])
