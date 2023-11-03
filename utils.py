from ast import literal_eval
from datetime import datetime
import logging
from pydoc import locate

import matplotlib.pyplot as plt
import numpy as np
import resource
import torch
import torchvision as tv
import wandb

from timm.utils import dispatch_clip_grad


def set_memory_limit(memory_in_gb:float):
    memory = memory_in_gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


class ContinualScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, disable_amp):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not disable_amp)

    def __call__(
        self, loss, optimizer, model_without_ddp, clip_grad=None, clip_mode='norm',
        parameters=None, create_graph=False,
        hook=True
    ):
        self.pre_step(loss, optimizer, parameters, create_graph, clip_grad, clip_mode)
        self.post_step(optimizer, model_without_ddp, hook)

    def pre_step(self, loss, optimizer, parameters=None, create_graph=False, clip_grad=None, clip_mode='norm'):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        if clip_grad is not None:
            assert parameters is not None
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

    def post_step(self, optimizer, model_without_ddp, hook=True):
        if hook and hasattr(model_without_ddp, 'hook_before_update'):
            model_without_ddp.hook_before_update()

        self._scaler.step(optimizer)

        if hook and hasattr(model_without_ddp, 'hook_after_update'):
            model_without_ddp.hook_after_update()

        self.update()

    def update(self):
        self._scaler.update()


def gin_config_to_dict(gin_config):
    params = {}
    for item in gin_config.replace("\\\n    ", "").split("\n"):
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
    logging.info(f"Saved model to {save_path}.")


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
