import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import gin
import gin.torch


@gin.configurable
class SupervisedTrainer:
    def __init__(self, model, device, optimizer=torch.optim.Adam, lr=0.001, wd=5e-4,
                 lr_scheduler=MultiStepLR, batch_size=64, train_loader=None):
        self.model = model
        self.device = device

        self.batch_size = batch_size
        self.optimizer = optimizer(self.model.parameters(), lr, weight_decay=wd)
        self.lr_scheduler = lr_scheduler(self.optimizer, milestones=[50, 100], gamma=0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.train_loader = train_loader


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
