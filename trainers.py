from abc import abstractmethod
import os

import gin
import gin.torch
import neptune
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import transforms as tfs

import losses
from utils import save_image


def trainer_maker(target_type, *args):
    print(f'\ntarget type: {target_type}\n')
    if target_type == 'supervised contrastive':
        return SupContrastiveTrainer(*args)
    elif target_type == 'supcon with simpreserving':
        return SimPresSupConTrainer(*args)
    elif target_type == 'supcon with interpolation':
        return InterpolSupConTrainer(*args)
    elif target_type == 'unsupervised contrastive':
        return UnsupContrastiveTrainer(*args)
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

                if (self.global_iters % self.log_interval == 0):
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
            print(f'\t Average accuracy after {self.current_task+1}: ', avg_accuracy)
            neptune.send_metric('avg accuracy', x=self.global_iters, y=avg_accuracy)
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupContrastiveTrainer(Trainer):
    CONTRAST_TYPES = ['simple', 'with_replay']

    def __init__(self, device, model, data, logdir, contrast_type=gin.REQUIRED, separate_memories=False,
                 prototype_memory_size=1000, replay_memory_size=None, replay_batch_size=None,
                 prototypes_mean_reduction=True):
        print('Supervised contrastive trainer.')
        super().__init__(device, model, data, logdir)
        self.loss_function = losses.SupConLoss(reduction='none')
        self.view_transforms = self.create_view_transforms()
        self.contrast_type = contrast_type
        self.separate_memories = separate_memories
        self.prototypes_mean_reduction = prototypes_mean_reduction

        err_message = "Contrast type must be element of {}".format(self.CONTRAST_TYPES)
        assert (contrast_type in self.CONTRAST_TYPES) == True, err_message

        if 'with_replay' in self.contrast_type:
            print('Use replay from memory.')
            err_message = "Parameter value must be set in config file"
            assert (replay_memory_size is not None) == True, err_message
            assert (replay_batch_size is not None) == True, err_message
            self.replay_batch_size = replay_batch_size
            self.replay_memory = ReservoirMemory(image_shape=self.data.input_shape,
                                                 target_shape=(1,),
                                                 device=self.device,
                                                 size_limit=replay_memory_size)

        print('Use prototypes for prediction.')
        if not self.separate_memories and 'with_replay' in self.contrast_type:
            self.prototype_memory = self.replay_memory
        else:
            self.prototype_memory = ReservoirMemory(image_shape=self.data.input_shape,
                                                    target_shape=(1,),
                                                    device=self.device,
                                                    size_limit=prototype_memory_size)

        self.prototype_manager = PrototypeManager(self.model,
                                                  self.prototype_memory,
                                                  self.device,
                                                  self.data.num_classes[0],
                                                  reduce_to_mean=self.prototypes_mean_reduction)


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


    def calc_loss_on_batch(self, input_images, target):
        input_images = input_images.to(self.device)
        target = target.to(self.device)

        input_images_combined = input_images
        target_combined = target

        if 'with_replay' in self.contrast_type and not self.replay_memory.empty():
            replay_images, replay_target = self.replay_memory.get_samples(self.replay_batch_size)
            input_images_combined = torch.cat([input_images, replay_images], dim=0)
            replay_target = replay_target.squeeze()
            target_combined = torch.cat([target, replay_target], dim=0)

        input_images_combined_view = self.create_image_views(input_images_combined)
        out1 = self.model(input_images_combined).unsqueeze(1)
        out2 = self.model(input_images_combined_view).unsqueeze(1)
        model_output_combined = torch.cat([out1, out2], dim=1)

        loss_per_sample = self.loss_function(model_output_combined, target_combined)
        loss_mean = torch.mean(loss_per_sample)

        return loss_mean


    def test_on_batch(self, batch):
        input_images, target = batch[0], batch[1]
        loss_mean = self.calc_loss_on_batch(input_images, target)

        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model.forward_features(input_images)

        predictions = self.predict_with_prototypes(model_output)

        accuracy = torch.mean(torch.eq(predictions, target).float())
        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy}
        return results


    def train_on_batch(self, batch):
        self.optimizer.zero_grad()
        results = self.test_on_batch(batch)
        loss_value = results['total_loss_mean']
        loss_value.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        if 'with_replay' in self.contrast_type:
            self.replay_memory.on_batch_end(*batch)
        if self.separate_memories or 'with_replay' not in self.contrast_type:
            self.prototype_memory.on_batch_end(*batch)
        prot_results = self.prototype_manager.on_batch_end()

        results.update(prot_results)
        if self.task_end:
            self.prototype_manager.on_task_end()
        return results


    def predict_with_prototypes(self, model_output):
        class_prototypes, prototype_labels = self.prototype_manager.get_prototypes()

        class_prototypes_tiled = torch.repeat_interleave(class_prototypes.unsqueeze(0),
                                                         self.batch_size,
                                                         dim=0)
        model_output_tiled = torch.repeat_interleave(model_output.unsqueeze(1),
                                                     class_prototypes_tiled.size()[1],
                                                     dim=1)

        dists = torch.norm(model_output_tiled - class_prototypes_tiled, p=2, dim=-1)
        prediction_indices = torch.argmin(dists, dim=-1)
        predictions = prototype_labels[prediction_indices]

        return predictions


    def log_images(self, batch):
        train_images = self.data.inverse_normalize(batch[0]).permute(0, 2, 3, 1)
        for i in range(self.batch_size):
            neptune.log_image('train images',
                              train_images[i].detach().cpu().numpy(),
                              image_name=str(batch[1][i].detach().cpu().numpy()))
        if 'with_replay' in self.contrast_type and not self.replay_memory.empty():
            imgs, lbls = self.replay_memory.get_samples(self.batch_size)
            imgs = self.data.inverse_normalize(imgs).permute(0, 2, 3, 1)
            for i in range(self.batch_size):
                neptune.log_image('replay images',
                                  imgs[i].detach().cpu().numpy(),
                                  image_name=str(lbls[i].item()))
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SimPresSupConTrainer(SupContrastiveTrainer):

    def __init__(self, device, model, data, logdir, sim_pres_weight=1000):
        print('Supervised contrastive trainer using similarity preserving constraint.')
        super().__init__(device, model, data, logdir)
        self.sim_pres_weight = sim_pres_weight
        self.prototype_manager = SimPresPrototypeManager(self.model,
                                                         self.prototype_memory,
                                                         self.device,
                                                         self.data.num_classes[0],
                                                         reduce_to_mean=self.prototypes_mean_reduction)

    def calc_loss_on_batch(self, input_images, target):
        loss_mean = super().calc_loss_on_batch(input_images, target)

        sim_diff = self.prototype_manager.similarity_matrix_curr - self.prototype_manager.similarity_matrix_prev
        sim_pres_loss_mean = torch.mean((sim_diff)**2).squeeze()
        print(f'Classification loss mean: {loss_mean} \t rehearsal loss mean: {sim_pres_loss_mean}')
        loss_mean += self.sim_pres_weight * sim_pres_loss_mean
        return loss_mean


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class InterpolSupConTrainer(SupContrastiveTrainer):

    def __init__(self, device, model, data, logdir, interpol_loss_weight=1000, num_interpol_points=3, num_pairs=1,
                 sim_type='cos'):
        print('Supervised contrastive trainer using interpolation constraint.')
        super().__init__(device, model, data, logdir)
        self.interpol_loss_weight = interpol_loss_weight
        self.num_interpol_points = num_interpol_points
        self.num_pairs = num_pairs
        self.sim_type = sim_type
        print('Sim type: ', self.sim_type)

        assert 3 <= num_interpol_points, "Number of interpolation points must be greater than or equal to 3."

    def calc_loss_on_batch(self, input_images, target):
        loss_mean = super().calc_loss_on_batch(input_images, target)

        interpol_loss = []
        for i in range(self.num_pairs):
            start_img, end_img = self._get_image_pair(input_images)
            interpol_loss.append(self.calc_interpol_loss(start_img, end_img, sim_type=self.sim_type))
        interpol_loss_mean = torch.stack(interpol_loss).mean()
        print(f'Classification loss mean: {loss_mean} \t interpol loss mean: {interpol_loss_mean}')

        loss_mean += self.interpol_loss_weight * interpol_loss_mean
        return loss_mean

    def _get_image_pair(self, input_images):
        if not self.replay_memory.empty():
            img_pair, _ = self.prototype_memory.get_samples(2)
        else:
            img_pair = input_images[:2]
        return img_pair[0], img_pair[1]

    def calc_interpol_loss(self, start_image, end_image, sim_type='l2'):
        alphas = np.linspace(0, 1, self.num_interpol_points)
        interpol_imgs = torch.stack([i * start_image + (1-i) * end_image for i in alphas])
        interpol_imgs = interpol_imgs.to(self.device)
        interpol_imgs.requires_grad = True

        feats = self.model.forward_features(interpol_imgs)
        start_feat, intermediate_feats, end_feat = feats[0].unsqueeze(0), feats[1:-1].unsqueeze(0), feats[-1].unsqueeze(0)
        if sim_type == 'l2':
            interpol_feats = [i * start_feat + (1-i) * end_feat for i in alphas]
            interpol_loss_mean = torch.norm(torch.stack(interpol_feats) - intermediate_feats, p=2).mean()
        elif sim_type == 'cos':
            base_sim = F.cosine_similarity(start_feat, end_feat)
            interpol_sim_targets = torch.tensor([i * base_sim + (1-i) * 1 for i in alphas], device=self.device).detach()
            interpol_sims = F.cosine_similarity(start_feat, feats)
            interpol_loss_mean = torch.square(torch.sub(interpol_sim_targets, interpol_sims)).mean()

        return interpol_loss_mean


    def _calc_interpol_loss_cos(self, start_image, end_image, sim_type='cos'):
        alphas = np.linspace(0, 1, self.num_interpol_points)
        interpol_imgs = torch.stack([i * start_image + (1-i) * end_image for i in alphas])
        interpol_imgs = interpol_imgs.to(self.device)

        feats = self.model.forward_features(interpol_imgs)
        start_feat, end_feat = feats[0], feats[-1]
        base_sim = F.cosine_similarity(start_feat, end_feat, dim=-1)
        interpol_sim_targets = [i * base_sim + (1-i) * 1 for i in alphas]
        interpol_sims = [F.cosine_similarity(start_feat, feat[i]) for i in range(len(feats))]
        interpol_loss_mean = torch.sub(interpol_sim_targets, interpol_sims).mean()
        return interpol_loss_mean


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class UnsupContrastiveTrainer(SupContrastiveTrainer):
    CONTRAST_TYPES = ['simple', 'with_replay', 'barlow_twins', 'barlow_twins_with_replay']

    def __init__(self, device, model, data, logdir):
        super().__init__(device, model, data, logdir)
        if 'barlow_twins' in self.contrast_type:
            self.loss_function = losses.BarlowTwinsLoss(device=self.device,
                                                        batch_size=self.batch_size,
                                                        feat_dim=self.model.features_dim,
                                                        offdiag_weight=1,
                                                        scale=1)
        else:
            self.loss_function = losses.SupConLoss(reduction='none')


    def calc_loss_on_batch(self, input_images, target):
        input_images = input_images.to(self.device)
        input_images_combined = input_images

        if 'with_replay' in self.contrast_type and not self.replay_memory.empty():
            replay_images, replay_target = self.replay_memory.get_samples(self.replay_batch_size)
            input_images_combined = torch.cat([input_images, replay_images], dim=0)

        input_images_combined_view = self.create_image_views(input_images_combined)

        if 'barlow_twins' in self.contrast_type:
            out1 = self.model.forward_features(input_images_combined)
            out2 = self.model.forward_features(input_images_combined_view)
            loss_mean = self.loss_function(out1, out2)
        else:
            out1 = self.model(input_images_combined)
            out2 = self.model(input_images_combined_view)

            model_output_combined = torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim=1)

            loss_per_sample = self.loss_function(model_output_combined)
            loss_mean = torch.mean(loss_per_sample)

        return loss_mean


class PrototypeManager:
    def __init__(self, model, memory, device, num_classes, reduce_to_mean=True):
        self.model = model
        self.memory = memory
        self.device = device
        self.num_classes = num_classes
        self.reduce_to_mean = reduce_to_mean

        self.prototypes = torch.zeros((self.num_classes, self.model.features_dim),
                                      requires_grad=False,
                                      device=self.device)
        self.prototype_labels = torch.tensor(range(self.num_classes),
                                             requires_grad=False,
                                             device=self.device)
        self._prev_prototypes = [self.prototypes]


    def _update_prototypes(self):
        print('Update prototypes.')
        p = []
        pl = []

        self.model.eval()
        with torch.no_grad():
            for label, images in self.memory.items_by_targets().items():
                features = self.model.forward_features(images).detach()
                if self.reduce_to_mean:
                    features = torch.mean(features, dim=0, keepdims=True)
                features = features / features.norm()
                p.append(features)
                pl.extend([label]*features.size(0))

        self.prototypes = torch.cat(p, dim=0)
        self.prototype_labels = torch.tensor(pl, requires_grad=False, device=self.device)
        return

    def on_batch_end(self):
        self._update_prototypes()
        diffs_to_log = self._get_diffs()
        return diffs_to_log

    def on_task_end(self):
        self._retain_prototypes()

    def get_prototypes(self):
        return self.prototypes, self.prototype_labels

    def _retain_prototypes(self):
        self._prev_prototypes.append(self.prototypes)

    def _get_diffs(self):
        diffs = {}
        for idx, pp in enumerate(self._prev_prototypes):
            size = min(self.prototypes.size(0), pp.size(0))
            l2_dist = torch.norm(pp[:size] - self.prototypes[:size], p=2, dim=-1).mean()
            cos_sim = F.cosine_similarity(pp[:size], self.prototypes[:size], dim=-1).mean()
            diffs[f'prototypes_l2_dist_mean_from_{idx}th'] = l2_dist
            diffs[f'prototypes_cos_sim_mean_from_{idx}th'] = cos_sim
        return diffs


class SimPresPrototypeManager(PrototypeManager):
    def __init__(self, model, memory, device, num_classes, reduce_to_mean=True):
        super().__init__(model, memory, device, num_classes, reduce_to_mean)
        self.similarity_matrix_prev = self.calc_similarity_matrix(self.prototypes, self.prototypes)
        self.similarity_matrix_curr = self.calc_similarity_matrix(self.prototypes, self.prototypes)

    def calc_similarity_matrix(self, a, b, eps=1e-9):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        cos_sim = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_sim

    def _update_similarity_matrix(self):
        self.similarity_matrix_prev = self.similarity_matrix_curr
        self.similarity_matrix_curr = self.calc_similarity_matrix(self.prototypes, self.prototypes)
        size = min(self.similarity_matrix_curr.size(0), self.similarity_matrix_prev.size(0))
        self.similarity_matrix_prev = self.similarity_matrix_prev[:size, :size]
        self.similarity_matrix_curr = self.similarity_matrix_curr[:size, :size]

    def on_batch_end(self):
        self._update_prototypes()
        self._update_similarity_matrix()
        diffs_to_log = self._get_diffs()
        return diffs_to_log


class Memory:
    def __init__(self, image_shape, target_shape, device, size_limit):
        self.size_limit = size_limit
        self.size = 0
        self.target2indices = {}
        self.content = {
            'images': torch.zeros((self.size_limit, *image_shape), device=device),
            'targets': torch.zeros((self.size_limit, *target_shape), dtype=torch.int32, device=device)
        }


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


    def get_samples(self, sample_size, target=None):
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


class ReservoirMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit):
        super().__init__(image_shape, target_shape, device, size_limit)
        self.num_seen_images_in_stream = 0


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


class FixedMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit, size_limit_per_target):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.size_limit_per_target = size_limit_per_target
        self.size_per_target = {}


    def _update_with_item(self, update_image, update_target):
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
        if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
            idx = self.size
            self._update_content_at_idx(update_image, update_target, idx)
            self.size += 1
            self.size_per_target[target_value] += 1

