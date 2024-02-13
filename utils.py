from ast import literal_eval
from datetime import datetime
import logging
from pydoc import locate

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import wandb


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

def save_model(model, model_path, add_datetime=False):
    if add_datetime:
        now = datetime.now()
        dt_str = f"_datettime={now.strftime('%d%m%Y_%H%M%S')}"
        save_path = model_path + dt_str + '.torch'
    else:
        save_path = model_path + '.pt'
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved model to {save_path}.")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    logging.info(f"Loaded model from {model_path}.")
    return model

def compare_weights(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            logging.info(f"Models have different weights: {param1.name} and {param2.name} are different.")
            logging.info(f"param1: {param1}")
            logging.info(f"param2: {param2}")
            return False
    return True

def get_model_trainable_params(model):
    return torch.cat([param.clone().detach().view(-1) for param in model.parameters() if param.requires_grad])

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def add_wandb_log_prefixes(results_to_log_dict):
    new = {("loss/" + key) if "loss" in key else ("accuracy/" + key) if "accuracy" in key else key: value
            for key, value in results_to_log_dict.items()}
    return new