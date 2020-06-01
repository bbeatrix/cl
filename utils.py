from ast import literal_eval
from pydoc import locate

import neptune
import numpy as np
import torchvision as tv


def rotate_image(image_array, angle):
    if angle == 0:
       return image_array
    elif angle == 90:
       return np.flipud(np.transpose(image_array, (0,2,1))).copy()
    elif angle == 180:
       return np.fliplr(np.flipud(image_array)).copy()
    elif angle == 270:
       return np.transpose(np.flipud(image_array), (0,2,1)).copy()


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
