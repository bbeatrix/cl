from abc import abstractmethod
import os

import gin
import gin.torch
import neptune
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

import losses, memories
from utils import save_image, save_model


def trainer_maker(target_type, *args):
    import contrastive_trainers
    print(f'\ntarget type: {target_type}\n')
    if target_type == 'supervised':
        return SupTrainer(*args)
    elif target_type == 'supervised with forgetstats':
        return SupTrainerWForgetStats(*args)
    elif target_type == 'supervised with replay':
        return SupTrainerWReplay(*args)
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
                self.on_iter_end(batch, batch_results)
            self.on_task_end()
        return

    def on_iter_end(self, batch, batch_results):
        is_task_start_or_end_iter = self.iter_count < 5 or self.iter_count > self.iters_per_task - 5

        if (self.global_iters % self.log_interval == 0) or is_task_start_or_end_iter:
            self._log_images(batch)

            if self.logdir is not None:
                save_image(batch[0][:self.batch_size, :, :, :],
                            name='train_images',
                            iteration=self.global_iters,
                            filename=os.path.join(self.logdir, 'train_images.png'))

            neptune.send_metric('learning_rate',
                                x=self.global_iters,
                                y=self.optimizer.param_groups[0]['lr'])

            results_to_log = {'train_' + key: value for key, value in batch_results.items()
                              if torch.is_tensor(value) == True}

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
        return

    def on_task_end(self):
        save_model(self.model,
                    os.path.join(self.logdir,
                                f"model_task={self.current_task}_globaliter={self.global_iters}"))
        self.test()
        self._log_avg_accuracy()

    def test(self):
        with torch.no_grad():
            self.model.eval()
            for idx, current_test_loader in enumerate(self.test_loaders):
                test_results = None
                for test_batch_count, test_batch in enumerate(current_test_loader, start=0):

                    test_batch_results = self.test_on_batch(test_batch)

                    if test_results is None:
                        test_results = test_batch_results.copy()
                        test_results = {key: value for key, value in test_results.items()
                                        if torch.is_tensor(value) == True}
                    else:
                        for metric in test_results.keys():
                            test_results[metric] += test_batch_results[metric].data

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


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupTrainerWForgetStats(SupTrainer):
    def __init__(self, device, model, data, logdir, log_score_freq=100):
        super().__init__(device, model, data, logdir)
        self.num_train_examples = len(data.train_dataset)
        self.forget_stats = {
            "prev_corrects": np.zeros(self.num_train_examples, dtype=np.int32),
            "num_forgets": np.zeros(self.num_train_examples, dtype=float),
            "never_correct": np.arange(self.num_train_examples, dtype=np.int32),
        }
        self.forget_scores = self.forget_stats["sum_forgets"].copy()
        self.forget_scores[self.forget_stats["never_correct"]] = np.inf
        self.log_score_freq = log_score_freq
        if not os.path.isdir(os.path.join(self.logdir, "forget_scores")):
            os.makedirs(os.path.join(self.logdir, "forget_scores"))

    def update_forget_stats(self, idxs, corrects):
        idxs_where_forgetting = idxs[self.forget_stats["prev_accs"][idxs] > corrects]
        self.forget_stats["num_forgets"][idxs_where_forgetting] += 1
        self.forget_stats["prev_corrects"][idxs] = corrects
        self.forget_stats["never_correct"] = np.setdiff1d(
            self.forget_stats["never_correct"],
            idxs[corrects.astype(bool)],
            True
        )
        self.forget_scores = self.forget_stats["num_forgets"].copy()
        self.forget_scores[self.forget_stats["never_correct"]] = np.inf
        return

    def save_forget_scores(self):
        save_path = os.path.join(self.logdir,
                                 "forget_scores",
                                 f"fs_task={self.current_task}_globaliter={self.global_iters}.npy")
        np.save(save_path, self.forget_scores)

    def test_on_batch(self, batch):
        input_images, target = batch[0], batch[1]
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        model_output = self.model(input_images)

        loss_mean = self.calc_loss_on_batch(model_output, target)
        predictions = torch.argmax(model_output, dim=1)
        corrects = torch.eq(predictions, target).detach().cpu().numpy().astype(int)
        accuracy_mean = torch.mean(torch.eq(predictions, target).float())

        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy_mean,
                   'corrects': corrects}
        return results

    def on_iter_end(self, batch, batch_results):
        super(SupTrainerWForgetStats, self).on_iter_end(batch, batch_results)
        indices_in_ds = batch[2]
        corrects = batch_results["corrects"]
        self.update_forget_stats(indices_in_ds, corrects)
        if self.global_iters % self.log_score_freq == 0:
            self.save_forget_scores()
        return

    def on_task_end(self):
        super(SupTrainerWForgetStats, self).on_task_end()
        self.save_forget_scores()
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupTrainerWReplay(SupTrainer):
    def __init__(self, device, model, data, logdir, use_replay=gin.REQUIRED,
                 replay_memory_size=None, replay_batch_size=None):
        print('Supervised trainer.')
        super().__init__(device, model, data, logdir)
        self.use_replay = use_replay

        if self.use_replay:
            print('Use replay from memory.')
            err_message = "Parameter value must be set in config file"
            assert (replay_memory_size is not None) == True, err_message
            assert (replay_batch_size is not None) == True, err_message
            self.replay_batch_size = replay_batch_size
            self.replay_memory = memories.ReservoirMemory(image_shape=self.data.input_shape,
                                                          target_shape=(1,),
                                                          device=self.device,
                                                          size_limit=replay_memory_size)

    def calc_loss_on_batch(self, input_images, target):
        input_images_combined = input_images
        target_combined = target

        if self.use_replay and not self.replay_memory.empty():
            replay_images, replay_target = self.replay_memory.get_samples(self.replay_batch_size)
            input_images_combined = torch.cat([input_images, replay_images], dim=0)
            replay_target = replay_target.squeeze()
            target_combined = torch.cat([target, replay_target], dim=0)

        model_output = self.model(input_images_combined)
        loss_per_sample = self.loss_function(model_output, target_combined)
        loss_mean = torch.mean(loss_per_sample)
        return loss_mean

    def test_on_batch(self, batch):
        input_images, target = batch[0], batch[1]
        input_images = input_images.to(self.device)
        target = target.to(self.device)
        loss_mean = self.calc_loss_on_batch(input_images, target)

        model_output = self.model(input_images)
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
        if self.use_replay:
            self.replay_memory.on_batch_end(*batch)
        return results
