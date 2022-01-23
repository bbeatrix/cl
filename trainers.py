from abc import abstractmethod
import os

import gin
import gin.torch
import neptune
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

import losses
from utils import save_image, save_model


def trainer_maker(target_type, *args):
    import contrastive_trainers
    print(f'\ntarget type: {target_type}\n')
    if target_type == 'supervised':
        return SupTrainer(*args)
    elif target_type == 'supervised contrastive':
        return contrastive_trainers.SupContrastiveTrainer(*args)
    elif target_type == 'supcon with simpreserving':
        return contrastive_trainers.SimPresSupConTrainer(*args)
    elif target_type == 'supcon with interpolation':
        return contrastive_trainers.InterpolSupConTrainer(*args)
    elif target_type == 'unsupervised contrastive':
        return contrastive_trainers.UnsupContrastiveTrainer(*args)
    else:
        raise NotImplementedError


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class Trainer:
    def __init__(self, device, model, data, logdir, log_interval=100, iters=gin.REQUIRED,
                 lr=gin.REQUIRED, wd=gin.REQUIRED, optimizer=gin.REQUIRED, lr_scheduler=MultiStepLR):
        self.device = device
        self.model = model
        self.data = data
        self.batch_size = data.batch_size
        self.num_tasks = data.num_tasks
        self.num_cycles = data.num_cycles
        self.train_loaders = data.loaders['train_loaders']
        self.test_loaders = data.loaders['test_loaders']
        self.log_interval = log_interval
        self.iters = iters
        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)

        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=[],
                                        gamma=0.1)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.logdir = logdir


    @abstractmethod
    def train_on_batch(self, batch):
        return

    @abstractmethod
    def test_on_batch(self, batch):
        return


    def train(self):
        self.global_iters = 0
        self.iters_per_task = self.iters // self.num_tasks // self.num_cycles
        print("Start training.")

        for self.current_task in range(0, self.num_tasks * self.num_cycles):
            self.task_end = False
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None

            for self.iter_count in range(1, self.iters_per_task + 1):
                self.model.train()
                self.global_iters += 1
                if self.iter_count == self.iters_per_task:
                    self.task_end = True

                try:
                    batch = next(current_train_loader_iterator)
                except StopIteration:
                    current_train_loader_iterator = iter(current_train_loader)
                    batch = next(current_train_loader_iterator)

                batch_results = self.train_on_batch(batch)

                is_task_start_or_end = self.iter_count < 5 or self.iter_count > self.iters_per_task - 5

                if (self.global_iters % self.log_interval == 0) or is_task_start_or_end:
                    self._log_images(batch)

                    if self.logdir is not None:
                        save_image(batch[0][:self.batch_size, :, :, :],
                                   name='train_images',
                                   iteration=self.global_iters,
                                   filename=os.path.join(self.logdir, 'train_images.png'))

                    neptune.send_metric('learning_rate',
                                        x=self.global_iters,
                                        y=self.optimizer.param_groups[0]['lr'])

                    results_to_log = {'train_' + key: value for key, value in batch_results.items()}

                    template = ("Task {}/{}x{}\tTrain\tglobal iter: {}, batch: {}/{}, metrics:  "
                                + "".join([key + ": {:.3f}  " for key in results_to_log.keys()]))
                    print(template.format(self.current_task + 1,
                                          self.num_tasks,
                                          self.num_cycles,
                                          self.global_iters,
                                          self.iter_count,
                                          self.iters_per_task,
                                          *[item.data for item in results_to_log.values()]))

                    for metric, result in results_to_log.items():
                        neptune.send_metric(metric, x=self.global_iters, y=result)

                    results_to_log = None

                    self.test()
            save_model(self.model,
                       os.path.join(self.logdir,
                                    f"model_task={self.current_task}_globaliter={self.global_iters}"))
            self.test()
            self._log_avg_accuracy()

        return


    def test(self):
        with torch.no_grad():
            self.model.eval()
            for idx, current_test_loader in enumerate(self.test_loaders):
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):

                    test_batch_results = self.test_on_batch(test_batch)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                    else:
                        for metric, result in test_batch_results.items():
                            test_results[metric] += result.data

                test_results = {f'task {idx+1} test_{key}': value / (test_batch_count + 1)
                                for key, value in test_results.items()}

                template = ("Task {}/{}x{}\tTest\tglobal iter: {} ({:.2f}%), metrics: "
                            + "".join([key + ": {:.3f}  " for key in test_results.keys()]))
                print(template.format(idx + 1,
                                      self.num_tasks,
                                      self.num_cycles,
                                      self.global_iters,
                                      float(self.global_iters) / self.iters * 100.,
                                      *[item.data for item in test_results.values()]))

                for metric, result in test_results.items():
                    neptune.send_metric(metric, x=self.global_iters, y=result)
        return


    def _log_images(self, batch):
        train_images = self.data.inverse_normalize(batch[0]).permute(0, 2, 3, 1)
        for i in range(self.batch_size):
            neptune.log_image('train images',
                              train_images[i].detach().cpu().numpy(),
                              image_name=str(batch[1][i].detach().cpu().numpy()))
        return


    def _log_avg_accuracy(self):
        with torch.no_grad():
            self.model.eval()
            avg_accuracy = 0
            for idx, current_test_loader in enumerate(self.test_loaders[:(self.current_task + 1)]):
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):
                    test_batch_results = self.test_on_batch(test_batch)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                    else:
                        for metric, result in test_batch_results.items():
                            test_results[metric] += result.data

                avg_accuracy += test_results['accuracy'] / (test_batch_count + 1)

            avg_accuracy /= (self.current_task + 1)
            print(f'\t Average accuracy after {self.current_task+1} task: ', avg_accuracy)
            neptune.send_metric('avg accuracy', x=self.global_iters, y=avg_accuracy)
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupTrainer(Trainer):
    def __init__(self, device, model, data, logdir):
        super().__init__(device, model, data, logdir)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def calc_loss_on_batch(self, model_output, target):
        loss_per_sample = self.loss_function(model_output, target)
        loss_mean = torch.mean(loss_per_sample)
        return loss_mean

    def test_on_batch(self, batch):
        input_images, target = batch[0], batch[1]
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)

        loss_mean = self.calc_loss_on_batch(model_output, target)
        predictions = torch.argmax(model_output, dim=1)
        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results

    def train_on_batch(self, batch):
        self.optimizer.zero_grad()
        results = self.test_on_batch(batch)
        results['total_loss_mean'].backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return results

