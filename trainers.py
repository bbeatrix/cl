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
        return ContrastiveTrainer(*args, multihead=False)
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
                 num_anchor_img_per_class=0, contrast_type=gin.REQUIRED, num_replay_images=0, num_anchor_images_total=200):
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
        self.num_replay_images = num_replay_images
        self.num_anchor_images_total = num_anchor_images_total

        CONTRAST_TYPES = ['with_replay', 'barlow_twins', 'similarity_rehearsal', 'simple']
        err_message = "Contrast type must be element of {}".format(CONTRAST_TYPES)
        assert (contrast_type in CONTRAST_TYPES) == True, err_message

        if self.contrast_type == 'with_replay':
            err_message = 'Contrastive learning with replay does not allow rehearsal yet!'
            assert self.rehearsal is False, err_message


class ReservoirMemory:
    def __init__(self, size_limit, image_shape, target_shape, device):
        self.size_limit = size_limit
        self.size = 0
        self.num_seen_images_in_stream = 0
        self.target2indices = {}
        self.content = {
            'images': torch.zeros((self.size_limit, *image_shape), device=device),
            'targets': torch.zeros((self.size_limit, *target_shape), dtype=torch.int32, device=device)
        }

    def get_content(self):
        return self.content['images'], self.content['targets']

    def on_batch_end(self, update_images, update_targets):
        for i in range(update_images.shape[0]):
            self._update_with_item(update_images[i], update_targets[i])


    def _update_content_at_idx(self, update_image, update_target, idx):
        self.content['images'][idx] = update_image
        self.content['targets'][idx] = update_target

        update_target_value = update_target.item()
        if update_target_value in self.target2indices.keys():
            self.target2indices[update_target_value].append(idx)
        else:
            self.target2indices[update_target_value] = [idx]


    def _remove_idx_with_target(self, idx, target):
        old_target = self.content['targets'][idx].item()
        self.target2indices[old_target].remove(idx)


    def _update_with_item(self, update_image, update_target):
        if self.size < self.size_limit:
            idx = self.size
            self._update_content_at_idx(update_image, update_target, idx)
            self.size += 1
        else:
            # memory is full.
            m = random.randrange(self.num_seen_images_in_stream)
            if m < self.size_limit:
                # Put it in
                idx = m
                self._remove_idx_with_target(idx, update_target)
                self._update_content_at_idx(update_image, update_target, idx)
        self.num_seen_images_in_stream += 1


    def get_samples(self, sample_size, target=None):
        #if target is None:
        #    selected_indices = random.choices(range(self.size), k=sample_size)
        #else:
        #    indices = self.target2indices[target]
        #    selected_indices = random.choices(indices, k=sample_size)

        if target is None:
            sample_size = min(self.size, sample_size)
            selected_indices = np.random.choice(range(self.size), sample_size)
        else:
            indices = self.target2indices[target]
            sample_size = min(sample_size, len(indices))
            selected_indices = np.random.choice(indices, sample_size)

        sample_images = self.content['images'][selected_indices]
        sample_targets = self.content['targets'][selected_indices]
        return sample_images, sample_targets


    def items_by_targets(self):
        result = {}
        for key, indices in self.target2indices.items():
            result[key] = self.content['images'][indices]
        return result


    def empty(self):
        return self.size == 0


class FixedMemory(ReservoirMemory):
    def __init__(self, size_limit, image_shape, target_shape, device, size_limit_per_target=200):
        self.size_limit = size_limit
        self.size_limit_per_target = size_limit_per_target
        self.size_per_target = {}
        self.size = 0
        self.target2indices = {}
        self.content = {
            'images': torch.zeros((self.size_limit, *image_shape), device=device),
            'targets': torch.zeros((self.size_limit, *target_shape), dtype=torch.int32, device=device)
        }

    def _update_with_item(self, update_image, update_target):
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
        if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
            idx = self.size
            self._update_content_at_idx(update_image, update_target, idx)
            self.size += 1
            self.size_per_target[target_value] += 1


class PrototypeManager:
    def __init__(self, model, memory, device, num_classes):
        self.model = model
        self.memory = memory
        self.device = device
        self.num_classes = num_classes

        self.prototypes = torch.zeros((self.num_classes, self.model.features_dim),
                                      requires_grad=False,
                                      device=self.device)
        self.prototype_labels = torch.tensor(range(self.num_classes),
                                             requires_grad=False,
                                             device=self.device)

    def update_prototypes(self):
        output_index = 0

        p = []
        pl = []

        self.model.eval()
        with torch.no_grad():
            for label, images in self.memory.items_by_targets().items():
                features = self.model.forward_features(images)[output_index].detach()
                features = torch.mean(features, dim=0, keepdims=True)
                features = features / features.norm()
                p.append(features)
                pl.extend([label]*features.size(0))

        self.prototypes = torch.cat(p, dim=0)
        self.prototype_labels = torch.tensor(pl, requires_grad=False, device=self.device)

    def get_prototypes(self):
        return self.prototypes, self.prototype_labels


class ClassMeanPrototypeManager(PrototypeManager):
    def __init__(self, model, memory, device):
        super().__init__(model, memory, device)

    def get_prototypes(self):
        pass

class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('CONTRAST TYPE: ', self.contrast_type)
        self.view_transforms = self.create_view_transforms()

        if self.rehearsal:
            print('Rehearsal is on.')
            self.similarity_matrix = None
            self.rehearsal_memory = ReservoirMemory(size_limit=1000, device=self.device)

        if self.use_prototypes:
            print('Use prototypes.')
            self.rehearsal_memory = ReservoirMemory(size_limit=1000, image_shape=self.data.input_shape, target_shape=(1,), device=self.device)
            self.prototype_manager = PrototypeManager(self.model, self.rehearsal_memory, self.device, self.data.num_classes[0])

    def init_similarity_matrix(self):
        if self.contrast_type == 'barlow_twins':
            self.similarity_matrix = torch.eye(self.data.num_classes[0] * self.num_anchor_img_per_class,
                                               requires_grad=False,
                                               device=self.device)
        else:
            self.similarity_matrix = None


    def get_batch_for_replay(self, num_replay_images):
        return self.rehearsal_memory.get_samples(num_replay_images)


    def create_view_transforms(self):
        view_transforms = [
            tfs.RandomResizedCrop(size=self.data.image_size, scale=(0.2, 1.)),
            tfs.RandomHorizontalFlip(),
            tfs.RandomApply([tfs.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            tfs.RandomGrayscale(p=0.2)]
        return tfs.Compose(view_transforms)


    def create_image_views(self, images, n_views=1):
        images_views = self.view_transforms(images)
        for _ in range(n_views - 1):
            new_view = self.view_transforms(images)
            images_views = torch.cat([images_views, new_view], dim=0)
        return images_views


    def calc_loss_on_batch(self, input_images, target, output_index=0):
        input_images = input_images.to(self.device)
        target = target.to(self.device)

        input_images_combined = input_images
        target_combined = target

        if self.contrast_type == 'with_replay' and not self.rehearsal_memory.empty():
            replay_images, replay_target = self.get_batch_for_replay(self.num_replay_images)
            input_images_combined = torch.cat([input_images, replay_images], dim=0)
            replay_target = replay_target.squeeze()
            target_combined = torch.cat([target, replay_target], dim=0)

        input_images_combined_view = self.create_image_views(input_images_combined)
        out1 = self.model(input_images_combined)[output_index].unsqueeze(1)
        out2 = self.model(input_images_combined_view)[output_index].unsqueeze(1)
        model_output_combined = torch.cat([out1, out2], dim=1)

        loss_per_sample = self.loss_function(model_output_combined, target_combined)
        loss_mean = torch.mean(loss_per_sample)

        if self.rehearsal and self.similarity_matrix is not None:
            print('Add rehearsal loss term.')
            if self.use_prototypes:
                current_prototypes, _ = self.prototype_manager.get_prototypes()
                prototypes = torch.stack(tuple(current_prototypes.values()), dim=0)
                prototypes_reshaped = prototypes.view(-1, prototypes.size(-1))
                current_similarity_matrix = self.calc_similarity_matrix(prototypes_reshaped)
                size = self.similarity_matrix.size(0)
                current_similarity_matrix = current_similarity_matrix[:size, :size]

            row = np.random.choice(current_similarity_matrix.shape[0], 1, replace=False)
            column = np.random.choice(current_similarity_matrix.shape[1], 1, replace=False)
            diff = self.similarity_matrix[row, column] - current_similarity_matrix[row, column]
            rehearsal_loss_mean = torch.mean((self.similarity_matrix - current_similarity_matrix)**2).squeeze()
            print(f'Classification loss mean: {loss_mean} \t rehearsal loss mean: {rehearsal_loss_mean}')
            loss_mean += self.rehearsal_weight * rehearsal_loss_mean

        return loss_mean


    def test_on_batch(self, input_images, target, output_index=0):
        loss_mean = self.calc_loss_on_batch(input_images, target, output_index)

        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model.forward_features(input_images)[output_index]

        if self.use_prototypes:
            predictions = self.predict_with_prototypes(model_output)
        else:
            predictions = torch.argmax(model_output, dim=1)

        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results

    def predict_with_prototypes(self, model_output, ncm=True):
        class_prototypes, prototype_labels = self.prototype_manager.get_prototypes()

        if ncm is True:
            class_prototypes_tiled = torch.repeat_interleave(class_prototypes.unsqueeze(0),
                                                             self.batch_size,
                                                             dim=0)
            model_output_tiled = torch.repeat_interleave(model_output.unsqueeze(1),
                                                         class_prototypes_tiled.size()[1],
                                                         dim=1)

            dists = torch.norm(model_output_tiled - class_prototypes_tiled, p=2, dim=-1)
            prediction_indices = torch.argmin(dists, dim=-1)
            predictions = prototype_labels[prediction_indices]
        else:
            class_prototypes = torch.stack(tuple(self.class_prototypes.values()), dim=0)
            class_prototypes_reshaped = class_prototypes.view(-1, class_prototypes.size(-1))
            prototype_similarities = self.calc_similarity_matrix(a=model_output,
                                                                 b=class_prototypes_reshaped)
            prediction_indices = torch.argmax(prototype_similarities, 1)
            predictions = prototype_labels[prediction_indices]
        return predictions



    def train_on_batch(self, batch, output_index=0):
        self.optimizer.zero_grad()
        results = self.test_on_batch(*batch, output_index)
        loss_value = results['total_loss_mean']
        loss_value.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.rehearsal_memory.on_batch_end(*batch)
        self.prototype_manager.update_prototypes()
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
                    self.log_images(batch)
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

            self.test()
            self.log_avg_accuracy()

        return

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
        return

    def log_images(self, batch):
        train_images = self.data.inverse_normalize(batch[0]).permute(0, 2, 3, 1)
        for i in range(self.batch_size):
            neptune.log_image('train images',
                              train_images[i].detach().cpu().numpy(),
                              image_name=str(batch[1][i].detach().cpu().numpy()))
        if self.rehearsal_memory is not None:
            imgs, lbls = self.rehearsal_memory.get_samples(self.batch_size)
            imgs = self.data.inverse_normalize(imgs).permute(0, 2, 3, 1)
            for i in range(self.batch_size):
                neptune.log_image('replay images',
                                  imgs[i].detach().cpu().numpy(),
                                  image_name=str(lbls[i].item()))
        return

    def log_avg_accuracy(self):
        with torch.no_grad():
            self.model.eval()
            avg_accuracy = 0
            for idx, current_test_loader in enumerate(self.test_loaders[:(self.current_task + 1)]):
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):
                    test_batch_results = self.test_on_batch(*test_batch)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                    else:
                        for metric, result in test_batch_results.items():
                            test_results[metric] += result.data

                avg_accuracy += test_results['accuracy'] / (test_batch_count + 1)

            avg_accuracy /= (self.current_task + 1)
            print(f'\t average accuracy: ', avg_accuracy)
            neptune.send_metric('avg accuracy', x=self.global_iters, y=avg_accuracy)
        return


    def calc_similarity_matrix(self, a, b=None, eps=1e-9):
        if b is None:
            b = a
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_sim

