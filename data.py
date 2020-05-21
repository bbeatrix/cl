import torch
from torchvision import datasets, transforms as tfs


class Data():
    def __init__(self, datadir, dataset_name, batch_size, target_type, dataloader_kwargs):
        self.datadir = datadir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self.target_type = target_type

        self._get_dataset()
        self._create_loaders()


    def _create_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        **self.dataloader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       **self.dataloader_kwargs)


    def _get_dataset(self):
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

            image_transforms = tfs.Compose([tfs.ToTensor(),
                                            tfs.Normalize(mean=[0.507, 0.487, 0.441],
                                                          std=[0.267, 0.256, 0.276])])

            self.train_dataset = datasets.CIFAR100(self.datadir,
                                                   train=True,
                                                   download=True,
                                                   transform=image_transforms)
            self.test_dataset = datasets.CIFAR100(self.datadir,
                                                  train=False,
                                                  download=True,
                                                  transform=image_transforms)

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
            raise Exception("Dataset not found: " + self.dataset_name)
