import logging
import gin
import gin.torch
import numpy as np
import os
import pickle
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms as tfs


@gin.configurable(denylist=['datadir', 'dataloader_kwargs'])
class Data:
    TARGET_TYPES = ['supervised']
    TASKS_SPLIT_TYPES = ["cl"]
    TASKS_ORDER = ["default_class_order"]
    INCREMENTAL_LEARNING_SETUP = ["CIL", "TIL"]

    def __init__(self, datadir, dataloader_kwargs, dataset_name='cifar10', image_size=32, batch_size=64,
                 target_type='supervised', augment=True, num_tasks=1, num_cycles=1,
                 apply_vit_transforms=False, simple_augmentation=False, normalization=False,
                 tasks_split_type="cl", tasks_order="default_class_order",
                 incremental_learning_setup="CIL", validation_ratio=0.0):

        err_message = "Data target type must be element of {}".format(self.TARGET_TYPES)
        assert (target_type in self.TARGET_TYPES) == True, err_message
        err_message = "Tasks' split type must be element of {}".format(self.TASKS_SPLIT_TYPES)
        assert (tasks_split_type in self.TASKS_SPLIT_TYPES) == True, err_message
        err_message = "Tasks' order type must be element of {}".format(self.TASKS_ORDER)
        assert (tasks_order in self.TASKS_ORDER) == True, err_message
        err_message = "Incremental learning type must be element of {}".format(self.INCREMENTAL_LEARNING_SETUP)
        assert (incremental_learning_setup in self.INCREMENTAL_LEARNING_SETUP) == True, err_message

        self.datadir = datadir
        self.dataloader_kwargs = dataloader_kwargs
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.target_type = target_type
        self.augment = augment
        self.num_tasks = num_tasks
        self.num_cycles = num_cycles
        self.apply_vit_transforms = apply_vit_transforms
        self.simple_augmentation = simple_augmentation
        self.normalization = normalization
        self.tasks_split_type = tasks_split_type
        self.tasks_order = tasks_order
        self.incremental_learning_setup = incremental_learning_setup
        self.validation_ratio = validation_ratio

        self._setup()

    @property
    def loaders(self):
        return {'train_loaders': self.train_loaders, 'test_loaders': self.test_loaders, 'valid_loaders': self.valid_loaders}

    def _setup(self):
        self._get_dataset()
        if os.path.isfile(
            f"{self.datadir}/{self.dataset_name}_{self.num_tasks}tasks_splittype={self.tasks_split_type}_validationratio={self.validation_ratio}_data.pkl"
        ):
            logging.info(f"Loading {self.dataset_name} {self.num_tasks} data from pickle.")
            with open(
                f"{self.datadir}/{self.dataset_name}_{self.num_tasks}tasks_splittype={self.tasks_split_type}_validationratio={self.validation_ratio}_data.pkl",
                "rb",
            ) as f:
                self.train_task_datasets = pickle.load(f)
                self.test_task_datasets = pickle.load(f)
                self.validation_task_datasets = pickle.load(f)
                self.train_task_datasets_indices_in_orig = pickle.load(f)
                self.test_task_datasets_indices_in_orig = pickle.load(f)
                self.validation_task_datasets_indices_in_orig = pickle.load(f)
                self.labels = pickle.load(f)
                self.num_classes_per_task = pickle.load(f)
                num_tasks = pickle.load(f)
                self.labels_per_task = pickle.load(f)
                assert (
                    num_tasks == self.num_tasks
                ), "Number of tasks in pickle file is not equal to the number of tasks."
            logging.info(f"Successfully loaded {self.dataset_name} {self.num_tasks} data from pickle.")
        else:
            self._create_tasks()
        self._create_loaders()

    def _get_dataset(self):
        logging.info(f"Loading {self.dataset_name} dataset from {self.datadir}.")

        image_transforms = [tfs.ToTensor()]
        self.inverse_normalize = torch.nn.Identity()
        augment_transforms = []

        if self.augment:
            logging.info("Using augmentation on train dataset.")
            self.input_shape = (3, self.image_size, self.image_size)

            if self.apply_vit_transforms is True:
                augment_transforms = [tfs.Resize(self.image_size),
                                      tfs.RandomHorizontalFlip()]

                image_transforms = [tfs.Resize(self.image_size),
                                    tfs.ToTensor(),
                                    tfs.Normalize(mean=[0.5, 0.5, 0.5],
                                                  std=[0.5, 0.5, 0.5])]
            if self.simple_augmentation:
                augment_transforms = [tfs.RandomCrop(self.image_size, padding=4),
                                      tfs.RandomHorizontalFlip()]
            else:
                augment_transforms = [tfs.RandomResizedCrop(size=self.image_size,
                                                            scale=(0.2, 1.)),
                                      tfs.RandomHorizontalFlip(),
                                      tfs.RandomApply([tfs.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                                      p=0.8),
                                      tfs.RandomGrayscale(p=0.2)]

        train_transforms = augment_transforms + image_transforms
        test_transforms = image_transforms

        if self.dataset_name == 'cifar10':
            self.input_shape, self.num_classes = (3, 32, 32), 10

            if self.normalization:
                train_transforms.append(tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.247, 0.243, 0.261)))
                test_transforms.append(tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                     std=(0.247, 0.243, 0.261)))
                self.inverse_normalize = tfs.Normalize(mean=(-0.4914/0.247, -0.4822/0.243, -0.4465/0.261),
                                                       std=[1/0.247, 1/0.243, 1/0.261])
            self.train_dataset = datasets.CIFAR10(self.datadir,
                                                  train=True,
                                                  download=True,
                                                  transform=tfs.Compose(train_transforms))
            self.test_dataset = datasets.CIFAR10(self.datadir,
                                                 train=False,
                                                 download=True,
                                                 transform=tfs.Compose(test_transforms))
        elif self.dataset_name == 'cifar100':
            self.input_shape, self.num_classes = (3, 32, 32), 100

            if self.normalization:
                train_transforms.append(tfs.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                                      std=(0.2673, 0.2564, 0.2762)))
                test_transforms.append(tfs.Normalize(mean=(0.5071, 0.4865, 0.4409),
                                                     std=(0.2673, 0.2564, 0.2762)))
                self.inverse_normalize = tfs.Normalize(mean=(-0.5071/0.2673, -0.4865/0.2564, -0.4409/0.2762),
                                                       std=[1/0.2673, 1/0.2564, 1/0.2762])
            self.train_dataset = datasets.CIFAR100(self.datadir,
                                                   train=True,
                                                   download=True,
                                                   transform=tfs.Compose(train_transforms))
            self.test_dataset = datasets.CIFAR100(self.datadir,
                                                  train=False,
                                                  download=True,
                                                  transform=tfs.Compose(test_transforms))
        elif self.dataset_name == 'miniimagenet':
            self.input_shape, self.num_classes = (3, 84, 84), 100

            if self.normalization:
                train_transforms.append(tfs.Normalize(mean=(0.4729, 0.4487, 0.4030),
                                                      std=(0.2833, 0.2752, 0.2886)))
                test_transforms.append(tfs.Normalize(mean=(0.4729, 0.4487, 0.4030),
                                                     std=(0.2833, 0.2752, 0.2886)))
                self.inverse_normalize = tfs.Normalize(mean=(-0.4729/0.2833, -0.4487/0.2752, -0.4030/0.2886),
                                                       std=[1/0.2833, 1/0.2752, 1/0.2886])

            train_in = open(self.datadir + "/miniimagenet/mini-imagenet-cache-train.pkl", "rb")
            train = pickle.load(train_in)
            train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
            val_in = open(self.datadir + "/miniimagenet/mini-imagenet-cache-val.pkl", "rb")
            val = pickle.load(val_in)
            val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
            test_in = open(self.datadir + "/miniimagenet/mini-imagenet-cache-test.pkl", "rb")
            test = pickle.load(test_in)
            test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
            all_data = np.vstack((train_x, val_x, test_x))
            all_data = np.transpose(all_data, (0, 1, 4, 2, 3)) / 255.0
            
            train_data, train_label, test_data, test_label = [], [], [], []

            for i in range(len(all_data)):
                class_x = all_data[i]
                class_y = np.ones((600,)) * i
                x_train = class_x[:-100]
                y_train = class_y[:-100]
                x_test = class_x[-100:]
                y_test = class_y[-100:]

                train_data.append(x_train)
                train_label.append(y_train)
                test_data.append(x_test)
                test_label.append(y_test)
            train_data = np.concatenate(train_data)
            train_label = np.concatenate(train_label)
            test_data = np.concatenate(test_data)
            test_label = np.concatenate(test_label)

            self.train_dataset = TensorDatasetWTransforms(train_data, train_label, tfs.Compose([tfs.ToPILImage()] + train_transforms))
            self.test_dataset = TensorDatasetWTransforms(test_data, test_label, tfs.Compose([tfs.ToPILImage()] + test_transforms))

        else:
            raise Exception("{} dataset not found!".format(self.dataset_name))

    def _create_random_split_task_datasets(self):
        assert (self.num_tasks == 2), "With the random split 2 tasks are allowed only"
        logging.info(f"Splitting training dataset into {self.num_tasks} random parts.")
        err_message =  "Number of tarining examples should be divisible by the number of tasks."
        assert len(self.train_dataset) % self.num_tasks == 0, err_message

        indices_permutation = np.random.permutation(len(self.train_dataset))
        num_concurrent_indices = len(self.train_dataset) // self.num_tasks

        for i in range(0, len(self.train_dataset), num_concurrent_indices):
            split_indices = indices_permutation[i: i + num_concurrent_indices]

            train_ds_subset = torch.utils.data.Subset(self.train_dataset,
                                                      split_indices)
            self.train_task_datasets.append(train_ds_subset)
            self.test_task_datasets.append(self.test_dataset)
            self.train_task_datasets_indices_in_orig.append(split_indices)
        logging.info(f"Number of train tasks: {len(self.train_task_datasets)}")
        return

    def _create_tasks(self):
        self.train_task_datasets, self.test_task_datasets, self.validation_task_datasets = [], [], []
        self.train_task_datasets_indices_in_orig, self.test_task_datasets_indices_in_orig = [], []
        self.validation_task_datasets_indices_in_orig = []
        self.labels = np.array([i for i in range(self.num_classes)])
        self.num_classes_per_task = self.num_classes // self.num_tasks
        self.labels_per_task = []

        if self.num_tasks == 1:
            self.labels = np.array([i for i in range(self.num_classes)])
            self.train_task_datasets = [self.train_dataset]
            self.test_task_datasets = [self.test_dataset]
            self.train_task_datasets_indices_in_orig = [list(range(len(self.train_dataset)))]
            self.test_task_datasets_indices_in_orig = [list(range(len(self.test_dataset)))]
            self.labels_per_task = [self.labels]

        elif self.num_tasks > 1 and self.tasks_split_type in ["cl"]:
            logging.info(f"Splitting training and test datasets into {self.num_tasks} parts for cl.")
            train_targets = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            test_targets = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
            self.labels = np.unique(train_targets)

            err_message = "Targets are assumed to be integers from 0 up to number of classes."
            assert set(self.labels) == set(range(self.num_classes)), err_message
            err_message =  "Number of classes should be divisible by the number of tasks."
            assert self.num_classes % self.num_tasks == 0, err_message

            num_concurrent_labels = self.num_classes // self.num_tasks

            for i in range(0, self.num_classes, num_concurrent_labels):
                concurrent_labels = self.labels[i: i + num_concurrent_labels]
                self.labels_per_task.append(concurrent_labels)

                trainset_filtered_indices = np.where(np.isin(train_targets, concurrent_labels))[0]
                testset_filtered_indices = np.where(np.isin(test_targets, concurrent_labels))[0]
                train_ds_subset = torch.utils.data.Subset(self.train_dataset,
                                                          trainset_filtered_indices)
                test_ds_subset = torch.utils.data.Subset(self.test_dataset,
                                                         testset_filtered_indices)
                self.train_task_datasets.append(train_ds_subset)
                self.test_task_datasets.append(test_ds_subset)
                self.train_task_datasets_indices_in_orig.append(trainset_filtered_indices)
                self.test_task_datasets_indices_in_orig.append(testset_filtered_indices)

        if self.validation_ratio > 0.0:
            self._create_validation_task_datasets()

        logging.info(f"Number of train tasks: {len(self.train_task_datasets)}")
        err_message = "Number of train datasets and the number of test datasets should be equal."
        assert len(self.train_task_datasets) == len(self.test_task_datasets), err_message

        logging.info(f"Number of train examples per train task: {[len(ds) for ds in self.train_task_datasets]}")
        logging.info(f"Number of test examples per test task: {[len(ds) for ds in self.test_task_datasets]}")

        logging.info("Getting number of train examples per class per train task dataset")
        for idx, ds in enumerate(self.train_task_datasets):
            targets = [ds[i][1] for i in range(len(ds))]
            num_examples_per_class_per_ds = {c: targets.count(c) for c in self.labels}
            logging.info(f"Number of train examples per classes in {idx + 1}. train task: {num_examples_per_class_per_ds}")
        with open(
            f"{self.datadir}/{self.dataset_name}_{self.num_tasks}tasks_splittype={self.tasks_split_type}_validationratio={self.validation_ratio}_data.pkl",
            "wb",
        ) as file:
            pickle.dump(self.train_task_datasets, file)
            pickle.dump(self.test_task_datasets, file)
            pickle.dump(self.validation_task_datasets, file)
            pickle.dump(self.train_task_datasets_indices_in_orig, file)
            pickle.dump(self.test_task_datasets_indices_in_orig, file)
            pickle.dump(self.validation_task_datasets_indices_in_orig, file)
            pickle.dump(self.labels, file)
            pickle.dump(self.num_classes_per_task, file)
            pickle.dump(self.num_tasks, file)
            pickle.dump(self.labels_per_task, file)
        return

    def _create_validation_task_datasets(self):
        logging.info(f"Creating validation subsets of size {self.validation_ratio} from each training dataset.")

        for idx, ds in enumerate(self.train_task_datasets):
            err_message =  f"Length of {idx}. task dataset {len(ds)} should be greater than the subset size {self.validation_ratio * len(ds)}."
            assert len(ds) >= self.validation_ratio * len(ds), err_message

            indices_permutation = np.random.permutation(len(ds))
            limit_index = int(self.validation_ratio * len(ds))
            subset_indices = indices_permutation[:limit_index]
            remaining_indices = indices_permutation[limit_index:]

            self.validation_task_datasets.append(torch.utils.data.Subset(ds, subset_indices))
            self.validation_task_datasets_indices_in_orig.append(self.train_task_datasets_indices_in_orig[idx][subset_indices])
            self.train_task_datasets[idx] = torch.utils.data.Subset(ds,remaining_indices)
            self.train_task_datasets_indices_in_orig[idx] = self.train_task_datasets_indices_in_orig[idx][remaining_indices]
        return

    def _create_loaders(self):
        self.num_classes = (self.num_classes,)
        if self.incremental_learning_setup == "TIL":
            self.num_classes = (self.num_classes_per_task,) * self.num_tasks
        _collate_func = default_collate

        logging.info("Creating train, validation and test data loaders.")
        self.train_loaders, self.test_loaders, self.valid_loaders = [], [], []

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
        for ds in self.validation_task_datasets:
            self.valid_loaders.append(torch.utils.data.DataLoader(ds,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=True,
                                                                 collate_fn=_collate_func,
                                                                 **self.dataloader_kwargs))
        self.train_loaders = self.train_loaders * self.num_cycles
        self.test_loaders = self.test_loaders * self.num_cycles
        self.valid_loaders = self.valid_loaders * self.num_cycles


class TensorDatasetWTransforms(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.tensordataset = torch.utils.data.TensorDataset(torch.from_numpy(images).float(),
                                                            torch.from_numpy(labels).long())
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensordataset[index][0]

        if self.transform:
            x = self.transform(x)

        y = self.tensordataset[index][1]

        return x, y

    def __len__(self):
        return len(self.tensordataset)
