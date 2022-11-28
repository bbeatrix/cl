from ast import literal_eval
from datetime import datetime
from pydoc import locate

import numpy as np
import torch
import torchvision as tv
import wandb


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
    wandb.log({name: filename}, step=iteration)

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
