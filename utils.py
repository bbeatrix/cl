from ast import literal_eval
from datetime import datetime
from pydoc import locate

import neptune
import numpy as np
import torch
import torchvision as tv
import yaml


def gin_config_to_dict(gin_config):
    params = {}
    for item in gin_config.split("\n"):
        if ' = ' in item:
            key, value = item.split(' = ')
            if locate(value) is not None:
                value = locate(value)
            elif "@" not in value:
                value = literal_eval(value)
            params[key] = value
    return params


def save_image(image, name, iteration, filename, normalize=True):
    tv.utils.save_image(image, filename, normalize=normalize)
    neptune.log_image(name, x=iteration, y=filename)

def save_model(model, model_path):
    now = datetime.now()
    dt_str = f"_datettime={now.strftime('%d%m%Y_%H%M%S')}"
    save_path = model_path + dt_str + '.torch'
    torch.save(model.state_dict(), save_path)
    print("Saved model to {}.".format(save_path))


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed like attributes (as well as normally).
    Credits: https://aiida.readthedocs.io/projects/aiida-core/en/latest/_modules/aiida/common/extendeddicts.html#AttributeDict  # noqa
    """

    def __init__(self, dictionary):
        """
        Recursively turn the `dict` and all its nested dictionaries into `AttrDict` instance.
        """
        super().__init__()

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            else:
                self[key] = value

    def to_dict(self):
        """
        Convert the AttrDict back to a dictionary.
        Helpful to feed the configuration to generic functions
        which only accept primitive types
        """
        dict = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                dict[k] = v.to_dict()
            else:
                dict[k] = v
        return dict

    def __getattr__(self, key):
        """
        Read a key as an attribute.
        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            return self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __setattr__(self, key, value):
        """
        Set a key as an attribute.
        """
        self[key] = value

    def __delattr__(self, key):
        """
        Delete a key as an attribute.
        :raises AttributeError: if the attribute does not correspond to an existing key.
        """
        if key in self:
            del self[key]
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {key}."
            )

    def __getstate__(self):
        """
        Needed for pickling this class.
        """
        return self.__dict__.copy()

    def __setstate__(self, dictionary):
        """
        Needed for pickling this class.
        """
        self.__dict__.update(dictionary)

    def __deepcopy__(self, memo=None):
        """
        Deep copy.
        """
        from copy import deepcopy

        if memo is None:
            memo = {}
        retval = deepcopy(dict(self))
        return self.__class__(retval)

    def __dir__(self):
        return self.keys()


def convert_to_attrdict(cfg, cmdline_args=None, dump_config=True):
    """
    Given the user input Hydra Config, and some command line input options
    to override the config file:
    1. merge and override the command line options in the config
    2. Convert the Hydra OmegaConf to AttrDict structure to make it easy
       to access the keys in the config file
    3. Also check the config version used is compatible and supported in vissl.
       In future, we would want to support upgrading the old config versions if
       we make changes to the VISSL default config structure (deleting, renaming keys)
    4. We infer values of some parameters in the config file using the other
       parameter values.
    """
    from omegaconf import OmegaConf
    if cmdline_args:
        # convert the command line args to DictConfig
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # check the cfg has valid version
    check_cfg_version(cfg)

    # assert the config and infer
    config = cfg.config
    infer_and_assert_hydra_config(config, cfg.engine_name)

    if dump_config:
        yaml_output_file = f"{cfg.experiment.logdir}/train_config.yaml"
        with open(yaml_output_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    convert_fsdp_dtypes(config)
    return cfg, config