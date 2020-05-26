import matplotlib
matplotlib.use("Agg")
import seaborn as sns; sns.set(style="ticks", font_scale=1.4, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt; plt.style.use('seaborn-deep')

import torchvision.utils as tvutils
import neptune
from pydoc import locate
from ast import literal_eval
import gin, gin.torch

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


def save_image(x, name, it, filename, normalize=True):
    tvutils.save_image(x, filename, normalize=normalize)
    neptune.log_image(name, x=it, y=filename)
