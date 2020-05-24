import torch
from torchvision import datasets, transforms as tfs
import numpy as np

class Data():
    def __init__(self, datadir, dataset_name, batch_size, target_type, dataloader_kwargs):
        self.datadir = datadir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self.target_type = target_type

        self._get_dataset()

        if self.target_type == 'selfsupervised':
            print('Self-supervised task: predicting rotation of 0, 90, 180 or 270 degrees.')
            self.train_dataset = SelfsupDataset(self.train_dataset)
            self.test_dataset = SelfsupDataset(self.test_dataset)
            self.num_classes = 4

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


class SelfsupDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        image = np.asarray(self.base_dataset[idx][0])
        target = np.random.choice([0, 1, 2, 3])
        rotated_image = self._rotate_image(image, target*90)
        return (torch.Tensor(rotated_image), target)


    def _rotate_image(self, image, angle):
        if angle == 0:
            return image
        elif angle == 90:
            return np.flipud(np.transpose(image, (0,2,1))).copy()
        elif angle == 180:
            return np.fliplr(np.flipud(image)).copy()
        elif angle == 270:
            return np.transpose(np.flipud(image), (0,2,1)).copy()
