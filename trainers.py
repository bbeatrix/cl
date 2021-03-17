import os

import gin
import gin.torch
import neptune
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from losses import SupConLoss
from utils import save_image



def trainer_maker(target_type, *args):
    print(f'\ntarget type: {target_type}\n')
    if target_type == 'auxiliary selfsupervised':
        return SupervisedWithAuxTrainer(*args)
    elif target_type == 'supervised multihead':
        return SupervisedTrainer(*args, multihead=True)
    else:
        return SupervisedTrainer(*args, multihead=False)


@gin.configurable(denylist=['device', 'model', 'data', 'logdir', 'multihead'])
class Trainer:
    def __init__(self, device, model, data, logdir, multihead, log_interval=100, iters=1000, lr=0.1,
                 lr_warmup_steps=500, wd=5e-4, optimizer=torch.optim.SGD, lr_scheduler=OneCycleLR,
                 rehearsal=False, loss=torch.nn.CrossEntropyLoss, use_prototypes=False):
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
        self.multihead = multihead
        self.rehearsal = rehearsal
        self.use_prototypes = use_prototypes
        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)

        self.lr_scheduler = OneCycleLR(self.optimizer,
                                       max_lr=lr,
                                       pct_start=lr_warmup_steps / self.iters,
                                       total_steps=self.iters)
        self.loss_function = loss(reduction='none')
        self.logdir = logdir


class SupervisedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rehearsal:
            print('Rehearsal is on.')
            self.num_anchor_img_per_task = 5
            self.anchor_images = None
            self.similarity_matrix = np.array([])
        if self.use_prototypes:
            self.class_prototypes = {}
            for i in range(self.data.num_classes[0]):
                self.class_prototypes[i] = torch.ones(self.data.num_classes[0],
                                                      self.data.num_classes[0],
                                                      requires_grad=False,
                                                      device=self.device)


    def test_on_batch(self, input_images, target, output_index=0):
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)[output_index]
        if self.use_prototypes:
            model_output = model_output.view(self.batch_size, 1, -1)
        loss_per_sample = self.loss_function(model_output, target)
        loss_mean = torch.mean(loss_per_sample)

        if self.rehearsal and self.anchor_images is not None:
            print('Add rehearsal loss term.')
            if self.use_prototypes:
                current_prototypes = self.get_prototypes()
                current_similarity_matrix = self.calc_similarity_matrix(current_prototypes)
            else:
                current_similarity_matrix = self.calc_similarity_matrix()

            row = np.random.choice(current_similarity_matrix.shape[0], 1, replace=False)
            column = np.random.choice(current_similarity_matrix.shape[1], 1, replace=False)
            diff = self.similarity_matrix[row, column] - current_similarity_matrix[row, column]
            loss_mean += np.mean((diff)**2)

        if self.use_prototypes:
            print('Predict with prototypes.')
            model_output = model_output.view(self.batch_size, -1)
            prototype_similarities = self.calc_similarity_matrix(a=model_output,
                                                                 b=torch.cat(tuple(self.class_prototypes.values())))
            predictions = torch.argmax(prototype_similarities, 1)
        else:
            predictions = torch.argmax(model_output, dim=1)

        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results


    def train_on_batch(self, batch, output_index=0):
        self.optimizer.zero_grad()
        results = self.test_on_batch(*batch, output_index)
        results['total_loss_mean'].backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return results


    def train(self):
        self.global_iters = 0
        self.iters_per_task = self.iters // self.num_tasks // self.num_cycles
        print("Start training.")

        for self.current_task in range(0, self.num_tasks * self.num_cycles):
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None

            for self.iter_count in range(1, self.iters_per_task + 1):
                self.global_iters += 1

                try:
                    batch = next(current_train_loader_iterator)
                except StopIteration:
                    current_train_loader_iterator = iter(current_train_loader)
                    batch = next(current_train_loader_iterator)
                if self.multihead:
                    batch_results = self.train_on_batch(batch, self.current_task % self.num_tasks)
                else:
                    batch_results = self.train_on_batch(batch)

                if results_to_log is None:
                    results_to_log = batch_results.copy()
                else:
                    for metric, result in batch_results.items():
                        results_to_log[metric] += result.data

                if (self.global_iters % self.log_interval == 0):
                    neptune.send_metric('learning_rate',
                                        x=self.global_iters,
                                        y=self.optimizer.param_groups[0]['lr'])

                    if self.logdir is not None:
                        save_image(batch[0][:self.batch_size, :, :, :],
                                   name='train_images',
                                   iteration=self.global_iters,
                                   filename=os.path.join(self.logdir, 'train_images.png'))

                    results_to_log = {'train_' + key: value / self.log_interval
                                      for key, value in results_to_log.items()}

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

            if self.rehearsal and self.current_task < self.num_tasks:
                self.retain_anchor_images()
                if self.use_rehearsal:
                    self.similarity_matrix = self.calc_similarity_matrix(a=torch.cat(tuple(self.class_prototypes.values())))
                else:
                    self.similarity_matrix = self.calc_similarity_matrix()


    def test(self):
        with torch.no_grad():
            for idx, current_test_loader in enumerate(self.test_loaders):
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):
                    if self.multihead:
                        test_batch_results = self.test_on_batch(*test_batch, idx)
                    else:
                        test_batch_results = self.test_on_batch(*test_batch)

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


    def retain_anchor_images(self):
        print('Retain anchor images.')
        current_train_loader_iterator = iter(self.train_loaders[self.current_task])
        c = self.data.num_classes[0] // self.num_tasks
        current_labels = self.data.labels[c * self.current_task : c * (self.current_task + 1)]

        for label in current_labels:
            task_ds = self.data.train_task_datasets[self.current_task]
            task_targets = [task_ds[i][1] for i in range(len(task_ds))]
            filtered_indices = np.where(np.isin(task_targets, label))[0]
            selected_indices = filtered_indices[:self.num_anchor_img_per_class]
            new_anchor_imgs = torch.stack([task_ds[i][0] for i in selected_indices], 0).to(self.device)
            if self.anchor_images is None:
                self.anchor_images = {label: new_anchor_imgs}
            else:
                self.anchor_images[label] = new_anchor_imgs
            if self.use_prototypes:
                with torch.no_grad():
                    features = self.model.forward(new_anchor_imgs).detach()#.cpu().numpy()
                    self.class_prototypes[label] = torch.mean(features, dim=0)


    def calc_similarity_matrix(self, a=None, b=None, eps=1e-9):
        print('Calculate similarity matrix.')
        if a is None:
            with torch.no_grad():
                images = torch.cat(list(self.anchor_images.values()))
                a = self.model(images).detach()#.cpu()#.numpy()
        if b is None:
            b = a
        #cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_sim


    def get_current_prototypes(self):
        print('Get current prototypes.')
        prototypes = {}
        with torch.no_grad():
            for label, images in self.anchor_images.items():
                features = self.model.forward(images).detach()#.cpu().numpy()
                prototypes[label] = torch.mean(features, dim=0)
        return prototypes


class SupervisedWithAuxTrainer(SupervisedTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.aux_loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def test_on_batch(self, input_images, target, aux_images, aux_target):
        inputs = list(map(lambda x: x.to(self.device), [input_images, aux_images]))
        targets = list(map(lambda x: x.to(self.device), [target, aux_target]))

        outputs = []
        for idx, x in enumerate(inputs):
            model_outputs = self.model(x)
            outputs.append(model_outputs[idx])

        loss_per_sample = self.loss_function(outputs[0], targets[0])
        loss_mean = torch.mean(loss_per_sample)
        aux_loss_per_sample = self.aux_loss_function(outputs[1], targets[1])
        aux_loss_mean = torch.mean(aux_loss_per_sample)
        total_loss_mean = loss_mean + aux_loss_mean

        predictions = torch.argmax(outputs[0], dim=1)
        accuracy = torch.mean(torch.eq(predictions, targets[0]).float())
        aux_predictions = torch.argmax(outputs[1], dim=1)
        aux_accuracy = torch.mean(torch.eq(aux_predictions, targets[1]).float())

        results = {'total_loss_mean': total_loss_mean,
                   'loss_mean': loss_mean,
                   'aux_loss_mean': aux_loss_mean,
                   'accuracy': accuracy,
                   'aux_accuracy': aux_accuracy}
        return results
