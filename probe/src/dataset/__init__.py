import os
import re

import torch
import torchvision
from dotmap import DotMap
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress deprecation warning coming from torchvision
from torchvision import transforms as T

from src.dataset.celeba import CelebAGenerator
from src.dataset.transforms import transformer
from src.utils import config
from src.utils.config import get_config

import numpy as np


def get_root(dataset_type):
    conf = get_config('config/default.env')
    return conf.dataset_root.__dict__[dataset_type]


def get_n_classes_and_channels(str_dataset):
    print(str_dataset)
    datasets = get_datasets(str_dataset)
    n_channels = int(datasets['train'][0][0].shape[0])
    return len(datasets.train.classes), n_channels


def get_data_loaders(str_dataset, seed=None, **kwargs):
    datasets = get_datasets(str_dataset)

    if seed is not None:
        torch.manual_seed(seed)

    common_settings = dict(batch_size=32, num_workers=4, drop_last=True)
    common_settings.update(**kwargs)

    if config.debug:
        batch_size = kwargs.get('batch_size', 50)
        datasets.train = data_utils.Subset(datasets.train, torch.arange(3 * batch_size + 1))
        datasets.val = data_utils.Subset(datasets.val, torch.arange(3 * batch_size + 1))

    train = DataLoader(datasets.train, shuffle=True, **common_settings)
    val = DataLoader(datasets.val, **common_settings)

    data_loaders = DotMap({'train': train, 'val': val})
    return data_loaders


def get_datasets(str_dataset):
    str_dataset = str_dataset.lower()

    mathches = re.search('(.*)_task_([0-9]*)_of_([0-9]*)', str_dataset, re.IGNORECASE)
    if mathches is not None:
        num_tasks = int(mathches.group(3))
        task_idx = int(mathches.group(2))-1
        str_dataset = mathches.group(1)
    else:
        num_tasks = 1
        task_idx = 0

    if str_dataset == 'fashion':
        trans = transformer['mnist']
        return create_pytorch_datasets(torchvision.datasets.FashionMNIST, trans, num_tasks, task_idx)
    elif str_dataset == 'mnist':
        trans = transformer['mnist']
        return create_pytorch_datasets(torchvision.datasets.MNIST, trans, num_tasks, task_idx)
    elif str_dataset == 'cifar10':
        trans = transformer['cifar10']
        return create_pytorch_datasets(torchvision.datasets.CIFAR10, trans, num_tasks, task_idx)
    elif str_dataset == 'cifar100':
        trans = transformer['cifar100']
        return create_pytorch_datasets(torchvision.datasets.CIFAR100, trans, num_tasks, task_idx)
    elif str_dataset in ['celeba', 'celebagenerator']:
        trans = transformer['celeba']
        return create_celebalucid_datasets('celeba', trans)
    else:
        raise ValueError('Dataset {} is unknown.'.format(str_dataset))


def create_pytorch_datasets(dataset_func, trans, num_tasks=1, task_idx=0):
    root = get_root('pytorch')
    train_trans = T.Compose(trans[0])
    val_trans = T.Compose(trans[1])

    try:
        train = dataset_func(root, train=True, transform=train_trans)
        val = dataset_func(root, train=False, transform=val_trans)
    except:
        train = dataset_func(root, train=True, download=True, transform=train_trans)
        val = dataset_func(root, train=False, download=True, transform=val_trans)

    if num_tasks > 1:

        targets = [train[i][1] for i in range(len(train))]
        targets_val = [val[i][1] for i in range(len(val))]
 
        labels = np.unique(targets)
        num_classes = labels.shape[0]
        num_concurrent_labels = num_classes // num_tasks
        concurrent_labels = labels[task_idx*num_concurrent_labels: (task_idx+1)*num_concurrent_labels]

        print("Num tasks is ", num_tasks, ", task_idx: ", task_idx, ", concurrent_labels: ", concurrent_labels)

        concurrent_targets = np.isin(targets, concurrent_labels)
        filtered_indices = np.where(concurrent_targets)[0]
        train_ds_subset = torch.utils.data.Subset(train, filtered_indices)

        concurrent_targets_val = np.isin(targets_val, concurrent_labels)
        filtered_indices_val = np.where(concurrent_targets_val)[0]
        val_ds_subset = torch.utils.data.Subset(val, filtered_indices_val)

        return DotMap({'train': train_ds_subset, 'val': val_ds_subset})

    return DotMap({'train': train, 'val': val})


def create_celebalucid_datasets(data_name, trans):
    root = get_root(data_name)

    csv = {
        'train': os.path.join(root, 'train.csv'),
        'val': os.path.join(root, 'test.csv')
    }

    trans = {
        'train': T.Compose(trans[0]),
        'val': T.Compose(trans[1])
    }

    train = CelebAGenerator(csv['train'], trans['train'])
    val = CelebAGenerator(csv['val'], trans['val'])
    return DotMap({'train': train, 'val': val})
