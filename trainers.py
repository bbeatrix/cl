import gin
import gin.torch

import torch
import torch.nn as nn
from torch import optim
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import neptune


@gin.configurable
class SupervisedTrainer:
    def __init__(self, model, device, batch_size, train_loader, test_loader,
                 log_interval=100, epochs=10, optimizer=torch.optim.SGD, lr=0.001, wd=5e-4,
                 lr_scheduler=torch.optim.lr_scheduler.MultiStepLR):

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.log_interval = log_interval
        self.epochs = epochs
        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)
        self.lr_schedule = lr_scheduler(self.optimizer,
                                        milestones=[epochs//4, epochs//2],
                                        gamma=0.2)
        self.loss_function = nn.CrossEntropyLoss()


    def train_on_batch(self, input_images, target):
        self.optimizer.zero_grad()
        results = self.test_on_batch(input_images, target)
        results['loss'].backward()
        self.optimizer.step()
        return results


    def test_on_batch(self, input_images, target):
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)
        loss = self.loss_function(model_output, target)
        predictions = torch.argmax(model_output, dim=1)
        correct = (predictions == target).sum()
        accuracy = torch.tensor(float(correct.data) / self.batch_size)
        return {'loss': loss, 'accuracy': accuracy}


    def train(self):
        self.global_iters = 0
        for self.current_epoch in range(1, self.epochs + 1):
            for batch_idx, (x, y) in enumerate(self.train_loader, start=0):
                self.global_iters += 1
                batch_results = self.train_on_batch(x, y)
                if self.global_iters % self.log_interval == 0:
                    template = "Train\tglobal iter: {}, epoch: {}, batch: {}/{}, metrics:  "
                    template += ": {:.3f}  ".join(list(batch_results.keys()) + [''])
                    print(template.format(self.global_iters,
                                          self.current_epoch,
                                          batch_idx + 1,
                                          len(self.train_loader),
                                          *[item.data for item in batch_results.values()]))
                    for metric, result in batch_results.items():
                        neptune.send_metric('batch_' + metric, x=self.global_iters, y=result)

                if self.global_iters % (10 * self.log_interval) == 0:
                    self.test()
        self.test()


    def test(self):
        test_results = None

        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(self.test_loader, start=0):
                test_batch_results = self.test_on_batch(x_test, y_test)

                if test_results is None:
                    test_results = test_batch_results.copy()
                else:
                    for metric, result in test_batch_results.items():
                        test_results[metric] += result.data

        total = len(self.test_loader)
        mean_test_results = {key: value/total for key, value in test_results.items()}

        template = "Test\tEpoch: {} ({:.2f}%), Metrics:  "
        template += ": {:.3f}  ".join(list(mean_test_results.keys()) + [''])
        print(template.format(self.current_epoch,
                              float(self.current_epoch) / (self.epochs) * 100.,
                              *[item.data for item in mean_test_results.values()]))

        for metric, result in mean_test_results.items():
            neptune.send_metric('test_' + metric, x=self.global_iters, y=result)
