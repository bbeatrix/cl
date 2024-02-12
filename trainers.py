from abc import abstractmethod
import copy
import logging
import os
import time

import gin
import gin.torch
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import wandb

import grad_log
import utils


def trainer_maker(target_type, *args):
    logging.info(f'\nTarget type: {target_type}\n')
    if target_type == 'supervised':
        return SupTrainer(*args)
    else:
        raise NotImplementedError


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class Trainer:
    def __init__(self, device, model, data, logdir, log_interval=100, iters=gin.REQUIRED, epochs_per_task=None,
                 lr=gin.REQUIRED, wd=gin.REQUIRED, optimizer=gin.REQUIRED, lossfunction_class=torch.nn.CrossEntropyLoss,
                 test_on_trainsets=False, log_grad_stats=False, log_margin_stats=False):
        self.device = device
        self.model = model
        self.data = data
        self.batch_size = data.batch_size
        self.num_tasks = data.num_tasks
        self.num_cycles = data.num_cycles
        self.train_loaders = data.loaders['train_loaders']
        self.test_loaders = data.loaders['test_loaders']
        self.valid_loaders = data.loaders['valid_loaders']
        self.log_interval = log_interval
        self.iters = iters
        self.epochs_per_task = epochs_per_task
        self.test_on_trainsets = test_on_trainsets
        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)

        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=[],
                                        gamma=0.1)
        self.loss_function = lossfunction_class(reduction='none')
        logging.info(f'Trainer loss function class: {self.loss_function}')
        self.best_valid_loss = np.inf

        self.log_grad_stats = log_grad_stats
        self.log_margin_stats = log_margin_stats

        self.logdir = logdir
        if not os.path.isdir(os.path.join(self.logdir, "model_checkpoints")):
            os.makedirs(os.path.join(self.logdir, "model_checkpoints"))

        self.model_params_history = [utils.get_model_trainable_params(self.model)]

    @abstractmethod
    def train_on_batch(self, batch):
        return

    @abstractmethod
    def test_on_batch(self, batch):
        return

    def train(self):
        self.global_iters = 0
        self.iters_per_task = self.iters // self.num_tasks // self.num_cycles
        self.task_accuracies = {}
        logging.info("Start training.")

        for self.current_task in range(0, self.num_tasks * self.num_cycles):
            self.task_end = False
            if len(self.data.num_classes) > 1:
                self.train_output_idx = self.current_task % self.num_tasks
            else:
                self.train_output_idx = 0
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None
            if self.epochs_per_task is not None:
                epochs2iters_per_task = self.epochs_per_task * len(current_train_loader)
                err_message = f"{self.epochs_per_task} * {len(current_train_loader)} should be equal to {self.iters_per_task}"
                assert (epochs2iters_per_task == self.iters_per_task), err_message
                self.iters_per_task = epochs2iters_per_task
            self.epoch_count = 0
            self.training_stop = False
            self.best_model = None
            for self.iter_count in range(1, self.iters_per_task + 1):
                logging.info(f"Task {self.current_task + 1}x{self.num_cycles}x{self.num_tasks}, iter {self.iter_count}/{self.iters_per_task}")
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
                self.on_iter_end(batch, batch_results)
                if self.iter_count % len(current_train_loader) == 0:
                    self.epoch_count += 1
                    self.on_epoch_end()
                    if self.training_stop:
                        break
            self.on_task_end()
        return

    def on_iter_end(self, batch_results):
        current_params = utils.get_model_trainable_params(self.model)
        param_dists = [torch.norm(current_params - params) for params in self.model_params_history]
        wandb.log({f"paramdist/param dist from param{i}": param_dists[i] for i in range(len(self.model_params_history))}, step=self.global_iters)
    
        is_task_start_or_end_iter = self.iter_count < 10 or self.iter_count > self.iters_per_task - 10

        if (self.global_iters % self.log_interval == 0) or is_task_start_or_end_iter:

            wandb.log({'learning_rate': self.optimizer.param_groups[0]['lr']}, step=self.global_iters)

            results_to_log = {'train_' + key: value for key, value in batch_results.items()
                              if torch.is_tensor(value) == True}

            template = ("Task {}/{}x{}\tTrain\tglobal iter: {}, batch: {}/{}, metrics:  "
                        + "".join([key + ": {:.3f}  " for key in results_to_log.keys()]))
            logging.info(template.format(self.current_task + 1,
                                         self.num_tasks,
                                         self.num_cycles,
                                         self.global_iters,
                                         self.iter_count,
                                         self.iters_per_task,
                                         *[item.data for item in results_to_log.values()]))

            results_to_log = utils.add_wandb_log_prefixes(results_to_log)
            wandb.log({metric: result for metric, result in results_to_log.items()}, step=self.global_iters)
            results_to_log = None

            self.test(self.test_loaders)
        return

    def on_epoch_end(self):
        if self.test_on_trainsets is True:
                self.test(self.train_loaders, testing_on_trainsets=True)
        if len(self.valid_loaders) > 0 and self.valid_loaders is not None:
            validation_loss, validation_accuracy = self.test([self.valid_loaders[self.current_task]], testing_on_validsets=True)
            logging.info(f"Validation loss: {validation_loss}, validation accuracy: {validation_accuracy}")
            if validation_loss < self.best_valid_loss:
                self.best_model = copy.deepcopy(self.model.state_dict())
                lr_patience = 6
            else:
                lr_patience -= 1
                if patience <= 0:
                    self.lr /= 2
                    logging.info(f'new lr={self.lr}')
                    if self.lr < 1e-5:
                        logging.infoo('Training stopped due to early stopping')
                        self.training_stop = True
                    patience = 6
                    for param_group in self.optimizer.param_groups:
                        if (self.epoch_count == 1):
                            param_group['lr'] = self.lr
                        else:
                            param_group['lr'] /= 2
        logging.info(f"Epoch {self.iter_count // len(self.train_loaders[self.current_task])} ended.")

    def on_task_end(self):
        logging.info(f"Task {self.current_task + 1} ended.")
        if self.best_model is not None:
            logging.info(f"Loading back task {self.current_task} best model.")
            self.model.load_state_dict(copy.deepcopy(self.best_model))
        self.model_params_history.append(utils.get_model_trainable_params(self.model))
        #utils.save_model(self.model,
        #                 os.path.join(self.logdir,
        #                              "model_checkpoints",
        #                              f"model_seed={self.seed}_task={self.current_task}_globaliter={self.global_iters}"))
        self.test(self.test_loaders)
        if self.test_on_trainsets is True:
            self.test(self.train_loaders, testing_on_trainsets=True)
        self._log_avg_accuracy_and_forgetting()

        self.learnt_targets.extend(list(self.data.labels_per_task[self.current_task]))
        logging.info(f"Learnt targets: {self.learnt_targets}")

    def test(self, dataset_loaders, testing_on_trainsets=False, testing_on_validsets=False):
        with torch.no_grad():
            self.model.eval()
            for idx, current_test_loader in enumerate(dataset_loaders):
                if len(self.data.num_classes) > 1:
                    test_output_index = idx % self.num_tasks
                else:
                    test_output_index = 0
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):

                    test_batch_results = self.test_on_batch(test_batch, test_output_index)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                        test_results = {key: value for key, value in test_results.items()
                                        if torch.is_tensor(value) == True}
                    else:
                        for metric in test_results.keys():
                            test_results[metric] += test_batch_results[metric].data

                if testing_on_validsets:
                    return test_results["total_loss_mean"] / (test_batch_count + 1), test_results["accuracy"] / (test_batch_count + 1)
                elif testing_on_trainsets:
                    test_results = {f'task {idx+1} test_on_trainsets_{key}': value / (test_batch_count + 1)
                                    for key, value in test_results.items()}
                    
                    template = ("Task {}/{}x{}\tTest on trainsets\tglobal iter: {} ({:.2f}%), metrics: "
                                + "".join([key + ": {:.3f}  " for key in test_results.keys()]))
                else:
                    test_results = {f'task {idx+1} test_{key}': value / (test_batch_count + 1)
                                    for key, value in test_results.items()}
                    template = ("Task {}/{}x{}\tTest\tglobal iter: {} ({:.2f}%), metrics: "
                                + "".join([key + ": {:.3f}  " for key in test_results.keys()]))

                logging.info(template.format(idx + 1,
                                             self.num_tasks,
                                             self.num_cycles,
                                             self.global_iters,
                                             float(self.global_iters) / self.iters * 100.,
                                             *[item.data for item in test_results.values()]))

                test_results = utils.add_wandb_log_prefixes(test_results)
                wandb.log({metric: result for metric, result in test_results.items()}, step=self.global_iters)
        return

    def _log_avg_accuracy_and_forgetting(self):
        with torch.no_grad():
            self.model.eval()
            avg_accuracy = 0
            for idx, current_test_loader in enumerate(self.test_loaders[:(self.current_task + 1)]):
                if len(self.data.num_classes) > 1:
                    test_output_index = idx % self.num_tasks
                else:
                    test_output_index = 0
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):
                    test_batch_results = self.test_on_batch(test_batch, test_output_index)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                        test_results = {key: value for key, value in test_results.items()
                                        if torch.is_tensor(value) == True}
                    else:
                        for metric in test_results.keys():
                            test_results[metric] += test_batch_results[metric].data

                acc_on_task = test_results['accuracy'] / (test_batch_count + 1)
                if idx not in self.task_accuracies.keys():
                    self.task_accuracies[idx] = [acc_on_task.cpu().numpy()]
                else:
                    self.task_accuracies[idx].append(acc_on_task.cpu().numpy())
                avg_accuracy += acc_on_task

            avg_accuracy /= (self.current_task + 1)
            logging.info(f'\t Average accuracy after {self.current_task+1} task: {avg_accuracy}')
            wandb.log({'average accuracy': avg_accuracy}, step=self.global_iters)

            if self.current_task > 0:
                avg_forgetting = 0
                for prev_task in range(0, self.current_task):
                    f_prev_task = max(self.task_accuracies[prev_task][:-1]) - self.task_accuracies[prev_task][-1]
                avg_forgetting += f_prev_task
                avg_forgetting /= (self.current_task)
                logging.info(f'\t Average forgetting after {self.current_task+1} task: {avg_forgetting}')
                wandb.log({'average forgetting': avg_forgetting}, step=self.global_iters)

            task_acc_array = np.array([v for v in self.task_accuracies.values()], dtype=object)
            wandb.log({'task accuracies dict': task_acc_array}, step=self.global_iters)
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupTrainer(Trainer):
    def __init__(self, device, model, data, logdir):
        super().__init__(device, model, data, logdir)
        logging.info(f'Supervised trainer loss function class: {self.loss_function}')

    def calc_loss_on_batch(self, model_output, target):
        if isinstance(self.loss_function, nn.MSELoss):
            target = torch.nn.functional.one_hot(target, num_classes=model_output.shape[-1]).float()
        loss_per_sample = self.loss_function(model_output, target)
        loss_mean = torch.mean(loss_per_sample)
        return loss_mean

    def test_on_batch(self, batch, output_idx=0):
        input_images, target = batch[0], batch[1]
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images, output_idx)
        if len(self.data.num_classes) > 1:
            target = target % self.data.num_classes[output_idx]

        loss_mean = self.calc_loss_on_batch(model_output, target)
        predictions = torch.argmax(model_output, dim=1)
        accuracy = torch.mean(torch.eq(predictions, target).float())

        if self.log_grad_stats:
            grad_log.log_batch_grad_stats(self.model,
                                          self.loss_function,
                                          self.global_iters,
                                          input_images,
                                          target,
                                          batch_type="train",
                                          output_index=output_idx)
        if self.log_margin_stats:
            grad_log.log_margin_stats(self.global_iters,
                                      model_output[:input_images.shape[0]].detach().clone(),
                                      target,
                                      batch_type="train")

        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results

    def train_on_batch(self, batch):
        self.optimizer.zero_grad()
        results = self.test_on_batch(batch, self.train_output_idx)
        results['total_loss_mean'].backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return results
