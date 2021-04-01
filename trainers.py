import os

import gin
import gin.torch
import neptune
import numpy as np
import random
import torch
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
import torchvision
from torchvision import transforms as tfs

from losses import SupConLoss
from utils import save_image



def trainer_maker(target_type, *args):
    print(f'\ntarget type: {target_type}\n')
    if target_type == 'supervised contrastive':
        return ContrastiveTrainer(*args)
    elif target_type == 'auxiliary selfsupervised':
        return SupervisedWithAuxTrainer(*args)
    elif target_type == 'supervised multihead':
        return SupervisedTrainer(*args, multihead=True)
    else:
        return SupervisedTrainer(*args, multihead=False)


@gin.configurable(denylist=['device', 'model', 'data', 'logdir', 'multihead'])
class Trainer:
    def __init__(self, device, model, data, logdir, multihead, log_interval=100, iters=1000, lr=0.1,
                 lr_warmup_steps=500, wd=5e-4, optimizer=torch.optim.SGD, lr_scheduler='LR',
                 rehearsal=False, loss=torch.nn.CrossEntropyLoss, use_prototypes=False, rehearsal_weight=1,
                 num_anchor_img_per_class=0, contrast_type=gin.REQUIRED):
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
        if lr_scheduler == 'OneCycleLR':
            self.lr_scheduler = OneCycleLR(self.optimizer,
                                           max_lr=lr,
                                           pct_start=lr_warmup_steps / self.iters,
                                           total_steps=self.iters)
        else:
            self.lr_scheduler = MultiStepLR(self.optimizer,
                                            milestones=[],
                                            gamma=0.1)

        self.loss_function = loss(reduction='none')
        self.logdir = logdir
        self.num_anchor_img_per_class = num_anchor_img_per_class
        self.rehearsal_weight = rehearsal_weight
        self.contrast_type = contrast_type

        CONTRAST_TYPES = ['with_replay', 'barlow_twins', 'similarity_rehearsal']
        err_message = "Contrast type must be element of {}".format(CONTRAST_TYPES)
        assert (contrast_type in CONTRAST_TYPES) == True, err_message

        if self.contrast_type == 'with_replay':
            err_message = 'Contrastive learning with replay does not allow rehearsal yet!'
            assert self.rehearsal is False, err_message

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('CONTRAST TYPE: ', self.contrast_type)
        if self.rehearsal:
            print('Rehearsal is on.')
            self.similarity_matrix = None
        if self.use_prototypes:
            print('Use prototypes.')
            print('Anchor images per class: ', self.num_anchor_img_per_class)
            self.anchor_images = None
            self.init_class_prototypes_and_labels()

    def init_similarity_matrix(self):
        if self.contrast_type == 'barlow_twins':
            self.similarity_matrix = torch.eye(self.data.num_classes[0] * self.num_anchor_img_per_class,
                                               requires_grad=False,
                                               device=self.device)
        else:
            self.similarity_matrix = None

    def init_class_prototypes_and_labels(self):
        self.class_prototypes = {}
        for i in range(self.data.num_classes[0]):
            all_zeros = torch.zeros(self.model.output_shape[0],
                                    requires_grad=False,
                                    device=self.device)
            all_zeros[i] = 1
            self.class_prototypes[i] = all_zeros
        self.prototype_labels = torch.tensor(range(self.data.num_classes[0]),
                                             requires_grad=False,
                                             device=self.device)

    def create_image_view(images, n_views=1):
        augment_transforms = [tfs.RandomResizedCrop(size=self.data.image_size, scale=(0.2, 1.)),
                              tfs.RandomHorizontalFlip(),
                              tfs.RandomApply([tfs.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                                              p=0.8),
                              tfs.RandomGrayscale(p=0.2)]
        transformed_image = tfs.Compose(augment)(images)
        return transformed_images

    def get_batch_for_replay(self):
        all_images_for_replay = torch.stack(tuple(self.anchor_images.values()), dim=0)
        selected_targets = random.choices(list(self.anchor_images.keys()), self.batch_size)
        selected_indices = random.choices(range(0, self.num_anchor_img_per_class), self.batch_size)
        indices = list(zip(selected_targets, selected_indices))
        replay_images = torch.gather(all_images_for_replay, dim=0, index=indices)
        replay_target = torch.tensor(selected_targets)
        return replay_images, replay_target

    def stack_dict_values(self, source_dict)
        stacked = torch.stack(tuple(source_dict.values()), dim=0)
        reshaped = stacked.view(-1, stacked.size(-1))
        return reshaped

    def test_on_batch(self, input_images, target, output_index=0):
        input_images_doubled = torch.cat([input_images[0], input_images[1]], dim=0)
        num_input_images = self.batch_size
        if self.contrast_type == 'with_replay':
            replay_images, replay_target = self.get_batch_for_replay()
            replay_images_transformed = self.create_image_view(replay_images).to(self.device)
            input_images_doubled = torch.cat([input_images_doubled, replay_images, replay_images_transformed], dim=0)
            target = torch.cat([target, replay_target], dim=0)
            num_input_images += len(replay_images)
        input_images_doubled = input_images_doubled.to(self.device)
        target = target.to(self.device)

        model_output_doubled = self.model(input_images_doubled)[output_index]

        f1, f2 = torch.split(model_output_doubled, [num_input_images, num_input_images], dim=0)
        model_output_splitted = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_per_sample = self.loss_function(model_output_splitted, target)
        loss_mean = torch.mean(loss_per_sample)

        if self.rehearsal and self.similarity_matrix is not None:
            print('Add rehearsal loss term.')
            if self.use_prototypes:
                current_prototypes, _ = self.get_current_prototypes()
                prototypes = torch.stack(tuple(current_prototypes.values()), dim=0)
                prototypes_reshaped = prototypes.view(-1, prototypes.size(-1))
                current_similarity_matrix = self.calc_similarity_matrix(prototypes_reshaped)
                size = self.similarity_matrix.size(0)
                current_similarity_matrix = current_similarity_matrix[:size, :size]
            #else:
            #    current_similarity_matrix = self.calc_similarity_matrix()

            row = np.random.choice(current_similarity_matrix.shape[0], 1, replace=False)
            column = np.random.choice(current_similarity_matrix.shape[1], 1, replace=False)
            diff = self.similarity_matrix[row, column] - current_similarity_matrix[row, column]
            rehearsal_loss_mean = torch.mean((self.similarity_matrix - current_similarity_matrix)**2).squeeze()
            print(f'Classification loss mean: {loss_mean} \t rehearsal loss mean: {rehearsal_loss_mean}')
            loss_mean += self.rehearsal_weight * rehearsal_loss_mean

        if self.use_prototypes:
            class_prototypes = torch.stack(tuple(self.class_prototypes.values()), dim=0)
            class_prototypes_reshaped = class_prototypes.view(-1, class_prototypes.size(-1))
            prototype_similarities = self.calc_similarity_matrix(a=model_output,
                                                                 b=class_prototypes_reshaped)
            prediction_indices = torch.argmax(prototype_similarities, 1)
            predictions = self.prototype_labels[prediction_indices]
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
            self.retain_anchor_images()
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None

            for self.iter_count in range(1, self.iters_per_task + 1):
                self.model.train()
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

                    if self.use_prototypes:
                        self.class_prototypes, self.prototype_labels = self.get_current_prototypes()

            if self.rehearsal and self.current_task < self.num_tasks:
                if self.use_prototypes:
                    class_prototypes = torch.stack(tuple(self.class_prototypes.values()), dim=0)
                    class_prototypes_reshaped = class_prototypes.view(-1, class_prototypes.size(-1))
                    self.similarity_matrix = self.calc_similarity_matrix(a=class_prototypes_reshaped)
                else:
                    self.similarity_matrix = self.calc_similarity_matrix()


    def test(self):
        with torch.no_grad():
            self.model.eval()
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
        print(current_labels)

        for label in current_labels:
            print('label: ', label)
            task_ds = self.data.train_task_datasets[self.current_task]

            task_targets = [task_ds[i][1] for i in range(len(task_ds))]
            filtered_indices = np.where(np.isin(task_targets, label))[0]
            selected_indices = filtered_indices[:self.num_anchor_img_per_class]
            selected_images = tuple([task_ds[i][0][0] for i in selected_indices])
            new_anchor_imgs = torch.stack(selected_images, 0).to(self.device)
            if self.anchor_images is None:
                self.anchor_images = {label: new_anchor_imgs}
            else:
                self.anchor_images[label] = new_anchor_imgs


    def calc_similarity_matrix(self, a, b=None, eps=1e-9):
        if b is None:
            b = a
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_sim


    def get_current_prototypes(self, output_index=0):
        print('Get current prototypes.')
        self.model.eval()
        prototypes = {}
        prototype_labels = []
        with torch.no_grad():
            for label, images in self.anchor_images.items():
                features = self.model(images)[output_index].detach()
                prototypes[label] = features
                prototype_labels.extend([label]*len(features))
        prototype_labels = torch.tensor(prototype_labels, requires_grad=False, device=self.device)
        return prototypes, prototype_labels


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
