from absl import logging
import gin
import gin.torch
import numpy as np
import torch
from torchvision import datasets, transforms as tfs
from torch.utils.data.dataloader import default_collate

from utils import rotate_image


@gin.configurable(blacklist=['datadir', 'dataloader_kwargs'])
class Data():
    def __init__(self, datadir, dataloader_kwargs, dataset_name='cifar100', batch_size=64,
                 target_type='supervised', augment=False, num_tasks=gin.REQUIRED):
        self.datadir = datadir
        self.dataloader_kwargs = dataloader_kwargs
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.target_type = target_type
        self.augment = augment
        self.num_tasks = num_tasks

        self._setup()


    @property
    def loaders(self):
        return {'train_loader': self.train_loaders, 'test_loader': self.test_loaders}


    def _setup(self):
        self._get_dataset()
        self._create_tasks()
        self._create_loaders()

    def _get_dataset(self):
        logging.info("Loading {} dataset from {}.".format(self.dataset_name, self.datadir))
        augment_transforms = []
        image_transforms = tfs.Compose([tfs.ToTensor()])

        if self.dataset_name == 'omniglot':
            self.input_shape, self.num_classes = (1, 105, 105), 1623

            self.train_dataset = datasets.Omniglot(self.datadir,
                                                   background=True,
                                                   target_transform=None,
                                                   download=True,
                                                   transform=image_transforms)
            self.test_dataset = datasets.Omniglot(self.datadir,
                                                  background=False,
                                                  target_transform=None,
                                                  download=True,
                                                  transform=image_transforms)
        elif self.dataset_name == 'cifar100':
            self.input_shape, self.num_classes = (3, 32, 32), 100

            if self.augment:
                logging.info("Usinging augmentation on train dataset.")
                augment_transforms = [tfs.RandomCrop(32, padding=4),
                                      tfs.RandomHorizontalFlip()]
            image_transforms = [tfs.ToTensor(),
                                tfs.Normalize(mean=[0.507, 0.487, 0.441],
                                              std=[0.267, 0.256, 0.276])]
            train_transforms = tfs.Compose(augment_transforms + image_transforms)
            test_transforms = tfs.Compose(image_transforms)

            self.train_dataset = datasets.CIFAR100(self.datadir,
                                                   train=True,
                                                   download=True,
                                                   transform=train_transforms)
            self.test_dataset = datasets.CIFAR100(self.datadir,
                                                  train=False,
                                                  download=True,
                                                  transform=test_transforms)
        elif self.dataset_name == 'cifar10':
            self.input_shape, self.num_classes = (3, 32, 32), 10

            self.train_dataset = datasets.CIFAR10(self.datadir,
                                                  train=True,
                                                  download=True,
                                                  transform=image_transforms)
            self.test_dataset = datasets.CIFAR10(self.datadir,
                                                 train=False,
                                                 download=True,
                                                 transform=image_transforms)
        else:
            raise Exception("{} dataset not found!".format(self.dataset_name))

    def _create_tasks(self):
        self.train_task_datasets, self.test_task_datasets = [], []

        if self.num_tasks > 1:
            logging.info("Splitting training dataset into {} parts for cl.".format(self.num_tasks))
            logging.info("The test set remains the original one for each task.")
            targets = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            labels = np.unique(targets)

            err_message = "Targets are assumed to be integers from 0 up to number of classes."
            assert set(labels) == set(range(self.num_classes)), err_message
            err_message =  "Number of classes should be divisible by the number of tasks."
            assert self.num_classes % self.num_tasks == 0, err_message

            num_concurrent_labels = self.num_classes // self.num_tasks

            for i in range(0, self.num_classes, num_concurrent_labels):
                concurrent_labels = labels[i: i + num_concurrent_labels]

                filtered_indices = np.where(np.isin(targets, concurrent_labels))[0]
                train_ds_subset = torch.utils.data.Subset(self.train_dataset,
                                                          filtered_indices)
                self.train_task_datasets.append(train_ds_subset)
        else:
            self.train_task_datasets = [self.train_dataset]
        self.test_task_datasets = [self.test_dataset] * self.num_tasks

        err_message = "Number of train datasets and the number of test datasets should be equal."
        assert len(self.train_task_datasets) == len(self.test_task_datasets), err_message

    def _create_loaders(self):
        if self.target_type == 'selfsupervised':
            logging.info("Self-supervised task: predicting rotation of 0, 90, 180 and 270 degrees.")
            logging.warning("Batch size becomes 4x larger, that is {}.".format(self.batch_size * 4))
            self.num_classes = 4

            def _collate_func(batch):
                batch = default_collate(batch)
                err_message = "A batch must contain two tensors: images, labels."
                assert len(batch) == 2, err_message

                images = np.asarray(batch[0])
                new_images, new_targets = [], []
                for img in images:
                    for target in range(0, 4):
                        rotated_image = rotate_image(img, angle=90*target)
                        new_images.append(rotated_image)
                        new_targets.append(target)
                images_array = np.array(new_images)
                targets_array = np.array(new_targets)
                return (torch.Tensor(images_array), torch.LongTensor(targets_array))
        else:
            _collate_func = default_collate

        logging.info("Creating train and test data loaders.")
        self.train_loaders, self.test_loaders = [], []

        for ds in self.train_task_datasets:
            self.train_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  shuffle=True,
                                                                  collate_fn=_collate_func,
                                                                  **self.dataloader_kwargs))
        for ds in self.test_task_datasets:
            self.test_loaders.append(torch.utils.data.DataLoader(ds,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 collate_fn=_collate_func,
                                                                 **self.dataloader_kwargs))
