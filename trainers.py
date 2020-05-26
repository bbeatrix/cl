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
                 log_interval=float("inf"), epochs=10, lr=0.001, wd=5e-4, optimizer=SGD,
                 lr_scheduler=MultiStepLR):

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_tasks = num_tasks

        self.log_interval = log_interval
        self.epochs = epochs
        self.total_iters = sum([len(loader) for loader in train_loader]) * self.epochs

        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)
        self.lr_schedule = lr_scheduler(self.optimizer,
                                        milestones=[epochs//4, epochs//2],
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
        self.global_iters = 0

        for self.current_task in range(0, self.num_tasks):
            current_train_loader = self.train_loader[self.current_task]
            iters_per_epoch = len(current_train_loader)

            for self.current_epoch in range(1, self.epochs + 1):
                results = None
                epoch_end = False
                for batch_idx, (x, y) in enumerate(current_train_loader, start=0):
                    self.global_iters += 1
                    results_log = None
                    batch_results = self.train_on_batch(x, y)
                    if results is None:
                        results = batch_results.copy()
                    else:
                        for metric, result in batch_results.items():
                            results[metric] += result.data

                    if (self.global_iters % self.log_interval == 0):
                        results_log = {'batch_'+ key: value
                                       for key, value in batch_results.items()}
                    elif iters_per_epoch == batch_idx + 1:
                        epoch_end = True
                        results_log = {'train_' + key: value/iters_per_epoch
                                       for key, value in results.items()}

                    if results_log is not None:
                        template = ("Task {}/{}\tTrain\t" +
                                    "global iter: {}, epoch: {}/{}, batch: {}/{}, metrics: " +
                                    ": {:.3f}  ".join(list(results_log.keys()) + ['']))
                        print(template.format(self.current_task + 1,
                                              self.num_tasks,
                                              self.global_iters,
                                              self.current_epoch,
                                              self.epochs,
                                              batch_idx + 1,
                                              iters_per_epoch,
                                              *[item.data for item in results_log.values()]))

                        for metric, result in results_log.items():
                            neptune.send_metric(metric, x=self.global_iters, y=result)

                        if epoch_end:
                            self.test()
                    else:
                        continue


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
                    "global iter: {} ({:.2f}%), epoch: {}/{} , metrics: " +
                    ": {:.3f}  ".join(list(test_results.keys()) + ['']))
        print(template.format(self.current_task + 1,
                              self.num_tasks,
                              self.global_iters,
                              float(self.global_iters)/self.total_iters * 100.,
                              self.current_epoch,
                              self.epochs,
                              *[item.data for item in test_results.values()]))

        for metric, result in test_results.items():
            neptune.send_metric(metric, x=self.global_iters, y=result)

