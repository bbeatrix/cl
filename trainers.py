import gin
import gin.torch

import torch
import torch.nn as nn
from torch import optim
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import neptune


@gin.configurable
class SupervisedTrainer:
    def __init__(self, model, device, batch_size, train_loader, test_loader, num_tasks,
                 log_interval=float("inf"), iters=10000, lr=0.001, wd=5e-4,
                 optimizer=SGD, lr_scheduler=MultiStepLR):

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_tasks = num_tasks
        self.log_interval = log_interval
        self.iters = iters

        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)
        self.lr_schedule = lr_scheduler(self.optimizer,
                                        milestones=[iters//4, iters//2],
                                        gamma=0.2)
        self.loss_function = nn.CrossEntropyLoss(reduction='none')


    def train_on_batch(self, input_images, target):
        self.optimizer.zero_grad()
        results = self.test_on_batch(input_images, target)
        results['loss_mean'].backward()
        self.optimizer.step()
        return results


    def test_on_batch(self, input_images, target):
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)
        loss_per_sample = self.loss_function(model_output, target)
        loss_mean = torch.mean(loss_per_sample)
        predictions = torch.argmax(model_output, dim=1)
        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results


    def train(self):
        print('Start training.')
        self.global_iters = 0
        self.iters_per_task = self.iters // self.num_tasks

        for self.current_task in range(0, self.num_tasks):
            current_train_loader = self.train_loader[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None

            for self.iter_count in range(1, self.iters_per_task + 1):
                self.global_iters += 1

                try:
                    (image_batch, target_batch) = next(current_train_loader_iterator)
                except StopIteration:
                    current_train_loader_iterator = iter(current_train_loader)
                    (image_batch, target_batch) = next(current_train_loader_iterator)

                batch_results = self.train_on_batch(image_batch, target_batch)

                if results_to_log is None:
                    results_to_log = batch_results.copy()
                else:
                    for metric, result in batch_results.items():
                        results_to_log[metric] += result.data

                if (self.global_iters % self.log_interval == 0):
                    results_to_log = {'train_'+ key: value / self.log_interval
                                      for key, value in results_to_log.items()}

                    template = ("Task {}/{}\tTrain\t" +
                                "global iter: {}, batch: {}/{}, metrics:  " +
                                ": {:.3f}  ".join(list(results_to_log.keys()) + ['']))
                    print(template.format(self.current_task + 1,
                                          self.num_tasks,
                                          self.global_iters,
                                          self.iter_count,
                                          self.iters_per_task,
                                          *[item.data for item in results_to_log.values()]))

                    for metric, result in results_to_log.items():
                        neptune.send_metric(metric, x=self.global_iters, y=result)

                    results_to_log = None
                    self.test()


    def test(self):
        test_results = None
        current_test_loader = self.test_loader[self.current_task]

        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(current_test_loader, start=0):
                test_batch_results = self.test_on_batch(x_test, y_test)

                if test_results is None:
                    test_results = test_batch_results.copy()
                else:
                    for metric, result in test_batch_results.items():
                        test_results[metric] += result.data

        test_results = {'test_' + key: value/(test_batch_idx + 1)
                        for key, value in test_results.items()}

        template = ("Task {}/{}\tTest\t" +
                    "global iter: {} ({:.2f}%), metrics: " +
                    ": {:.3f}  ".join(list(test_results.keys()) + ['']))
        print(template.format(self.current_task + 1,
                              self.num_tasks,
                              self.global_iters,
                              float(self.global_iters)/self.iters * 100.,
                              *[item.data for item in test_results.values()]))

        for metric, result in test_results.items():
            neptune.send_metric(metric, x=self.global_iters, y=result)
