from abc import abstractmethod
import logging
import os

import gin
import gin.torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import wandb

import losses, memories
import utils


def trainer_maker(target_type, *args):
    import contrastive_trainers
    logging.info(f'\nTarget type: {target_type}\n')
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
    def __init__(self, device, model, data, logdir, log_interval=100, iters=gin.REQUIRED, epochs_per_task=None,
                 lr=gin.REQUIRED, wd=gin.REQUIRED, optimizer=gin.REQUIRED, lr_scheduler=MultiStepLR, test_on_trainsets=False,
                 test_on_controlgroup=False):
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
        self.epochs_per_task = epochs_per_task
        self.test_on_trainsets = test_on_trainsets
        self.test_on_controlgroup = test_on_controlgroup
        self.optimizer = optimizer(self.model.parameters(),
                                   lr,
                                   weight_decay=wd)

        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=[],
                                        gamma=0.1)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.logdir = logdir
        if not os.path.isdir(os.path.join(self.logdir, "model_checkpoints")):
            os.makedirs(os.path.join(self.logdir, "model_checkpoints"))

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
            current_train_loader = self.train_loaders[self.current_task]
            current_train_loader_iterator = iter(current_train_loader)
            results_to_log = None
            if self.epochs_per_task is not None:
                epochs2iters_per_task = self.epochs_per_task * len(current_train_loader)
                err_message = f"{self.epochs_per_task} * {len(current_train_loader)} should be equal to {self.iters_per_task}"
                assert (epochs2iters_per_task == self.iters_per_task), err_message
                self.iters_per_task = epochs2iters_per_task
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
        if self.test_on_controlgroup:
            self._log_controlgroup_predictions()

        is_task_start_or_end_iter = self.iter_count < 10 or self.iter_count > self.iters_per_task - 10

        if (self.global_iters % self.log_interval == 0) or is_task_start_or_end_iter:
            self._log_train_images(batch)

            if self.logdir is not None:
                utils.save_image(batch[0][:self.batch_size, :, :, :],
                                 name='train_images',
                                 iteration=self.global_iters,
                                 filename=os.path.join(self.logdir, 'train_images.png'))

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

            wandb.log({metric: result for metric, result in results_to_log.items()}, step=self.global_iters)
            results_to_log = None

            self.test(self.test_loaders)
            if self.test_on_trainsets is True:
                self.test(self.train_loaders, testing_on_trainsets=True)
        return

    def on_task_end(self):
        utils.save_model(self.model,
                         os.path.join(self.logdir,
                                      "model_checkpoints",
                                      f"model_task={self.current_task}_globaliter={self.global_iters}"))
        self.test(self.test_loaders)
        if self.test_on_trainsets is True:
            self.test(self.train_loaders, testing_on_trainsets=True)
        self._log_avg_accuracy_and_forgetting()
        
        logging.info("Logging dist_from_mean_feats")
        if not os.path.isdir(os.path.join(self.logdir, "dist_from_mean_feats")):
            os.makedirs(os.path.join(self.logdir, "dist_from_mean_feats"))
        dist_from_mean_feats, normalized_dist_from_mean_feeats = self._get_dist_from_mean_feats(self.mean_feats,
                                                                                                self.normalized_mean_feats,
                                                                                                self.data.full_trainset_loader)
        np.savetxt(os.path.join(self.logdir, "dist_from_mean_feats", f"dist_from_mean_feats_task={self.current_task}_globaliter={self.global_iters}.npy"), dist_from_mean_feats)
        np.save(os.path.join(self.logdir, "dist_from_mean_feats", f"normalized_dist_from_mean_feats_task={self.current_task}_globaliter={self.global_iters}.npy"), normalized_dist_from_mean_feeats)
        logging.info("Task {} ended.".format(self.current_task + 1))

    def test(self, dataset_loaders, testing_on_trainsets=False):
        with torch.no_grad():
            self.model.eval()
            for idx, current_test_loader in enumerate(dataset_loaders):
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

                if testing_on_trainsets:
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

                wandb.log({metric: result for metric, result in test_results.items()}, step=self.global_iters)
        return

    def _log_train_images(self, batch):
        train_images = self.data.inverse_normalize(batch[0])
        image_grid = torchvision.utils.make_grid(train_images)
        ax = plt.imshow(np.transpose(image_grid, (1, 2, 0)))
        plt.axis('off')
        fig = plt.gcf()
        wandb.log({"train image batch": fig}, step=self.global_iters)
        plt.close()
        return

    def _log_avg_accuracy_and_forgetting(self):
        with torch.no_grad():
            self.model.eval()
            avg_accuracy = 0
            for idx, current_test_loader in enumerate(self.test_loaders[:(self.current_task + 1)]):
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
            if not os.path.isdir(os.path.join(self.logdir, "task_accuracies")):
                os.makedirs(os.path.join(self.logdir, "task_accuracies"))
            save_path = os.path.join(self.logdir,
                                    "task_accuracies",
                                    f"task_accuracies_after_task={self.current_task}_globaliter={self.global_iters}.txt")
            np.savetxt(save_path, np.array([np.array(v) for v in self.task_accuracies.values()]), delimiter=', ', fmt='%s')
        return

    def _log_controlgroup_predictions(self):
        logging.info("Logging info on control group images' predictions")
        controlgroup_images = torch.stack([item[0] for cgdict in self.data.control_group.values() for cgitemlist in cgdict.values() for item in cgitemlist]) # 3 x 3x10 the first dim? 
        
        controlgroup_images = controlgroup_images.to(self.device)
        controlgroup_model_outputs = self.model(controlgroup_images).detach().cpu()
        pred_dim = controlgroup_model_outputs.shape[-1]
        self.mean_feats, self.normalized_mean_feats = self._get_mean_features()

        log_dict = {}
        imgidx = 0
        for gkey, control_group_dict in self.data.control_group.items():
            for lidx, cgitemlist in enumerate(control_group_dict.values()):
                for itemidx in range(len(cgitemlist)):
                    right_pred = controlgroup_model_outputs[imgidx][lidx]
                    max_pred = max(controlgroup_model_outputs[imgidx])
                    softmax_pred = torch.nn.functional.softmax(controlgroup_model_outputs[imgidx], dim=-1)
                    onehot_pred = torch.nn.functional.one_hot(torch.tensor(lidx), num_classes=pred_dim)
                    el2n = torch.norm(softmax_pred - onehot_pred, p=2, dim=-1)
                    log_dict[f"cg el2n/{gkey}/cg target {lidx} {itemidx}. image el2n_score"] = el2n.numpy()

                    dist_from_mean_feat = torch.norm(self.mean_feats[lidx] - controlgroup_model_outputs[imgidx], p=2, dim=-1)
                    normalized_output = controlgroup_model_outputs[imgidx] / torch.norm(controlgroup_model_outputs[imgidx], p=2, dim=-1)
                    dist_from_normalized_mean_feat = torch.norm(self.normalized_mean_feats[lidx] - normalized_output, p=2, dim=-1)
                    log_dict[f"cg dist from mean feat/{gkey}/cg target {lidx} {itemidx}. image dist_from_mean_feat"] = dist_from_mean_feat.numpy()
                    log_dict[f"cg dist from normalized mean feat/{gkey}/cg target {lidx} {itemidx}. image dist_from_normalized_mean_feat"] = dist_from_normalized_mean_feat.numpy()

                    log_dict[f"cg pred diff/{gkey}/cg target {lidx} {itemidx}. image prediction diff from max"] = right_pred - max_pred
                    for c in range(pred_dim):
                        log_dict[f"cg pred/{gkey}/cg target {lidx} {itemidx}. image prediction {c}"] = controlgroup_model_outputs[imgidx][c]
                        log_dict[f"cg pred softmax/{gkey}/cg target {lidx} {itemidx}. image softmax prediction {c}"]= softmax_pred[c]
                    imgidx += 1

        wandb.log({k: v for k,v in log_dict.items()}, step=self.global_iters)
        images = self.data.inverse_normalize(controlgroup_images.detach().cpu())
        fig = plt.figure(figsize = (20, 40))
        image_grid = torchvision.utils.make_grid(images)
        ax = plt.imshow(np.transpose(image_grid, (1, 2, 0)))
        plt.axis('off')
        fig = plt.gcf()
        wandb.log({"control group images": fig}, step=self.global_iters)
        plt.close()
        return

    def _get_mean_features(self):
        mean_feats, normalized_mean_feats = {}, {}
        normalized_mean_feats = {}
        for target, images in self.data.images_per_targets.items():
            images = images.to(self.device)
            targetimages_model_outputs = self.model(images).detach().cpu()
            normalized_targetimages_model_outputs = targetimages_model_outputs / torch.norm(targetimages_model_outputs, p=2, dim=1, keepdim=True)
            mean_feats[target] = torch.mean(targetimages_model_outputs, dim=0)
            normalized_mean_feats[target] = torch.mean(normalized_targetimages_model_outputs, dim=0)
            normalized_mean_feats[target] /= torch.norm(normalized_mean_feats[target])
        return mean_feats, normalized_mean_feats
    
    def _get_dist_from_mean_feats(self, mean_feats, normalized_mean_feats, dataloader):
        dist_from_mean_feats, normalized_dist_from_mean_feats = torch.zeros(len(self.data.train_dataset)), torch.zeros(len(self.data.train_dataset))
        for batch in iter(dataloader):
            images, targets, indices = batch
            images = images.to(self.device)
            batch_model_outputs = self.model(images).detach().cpu()
            for idx, target in enumerate(targets):
                dist_from_mean_feats[indices[idx]] = torch.norm(batch_model_outputs[idx] - mean_feats[target.item()])
                normalized_dist_from_mean_feats[indices[idx]] = torch.norm(batch_model_outputs[idx] / torch.norm(batch_model_outputs[idx]) - normalized_mean_feats[target.item()])
        return dist_from_mean_feats, normalized_dist_from_mean_feats
        

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
            "first_learn_iters": np.inf * np.ones(self.num_train_examples, dtype=np.int32),
        }
        self.forget_scores = self.forget_stats["num_forgets"].copy()
        self.forget_scores[self.forget_stats["never_correct"]] = np.inf
        self.log_score_freq = log_score_freq
        if not os.path.isdir(os.path.join(self.logdir, "forget_scores")):
            os.makedirs(os.path.join(self.logdir, "forget_scores"))
        if not os.path.isdir(os.path.join(self.logdir, "first_learn_iters")):
            os.makedirs(os.path.join(self.logdir, "first_learn_iters"))

    def update_forget_stats(self, idxs, corrects):
        count_first_learns = 0
        for i, idx in enumerate(idxs):
            if self.forget_stats["first_learn_iters"][idx] == np.inf and corrects[i] == 1:
                self.forget_stats["first_learn_iters"][idx] = self.global_iters
                count_first_learns += 1
        wandb.log({"count first_learns in iter": count_first_learns}, step=self.global_iters)
        idxs_where_forgetting = idxs[self.forget_stats["prev_corrects"][idxs] > corrects]
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

    def save_first_learn_iters(self):
        save_path = os.path.join(self.logdir,
                                 "first_learn_iters",
                                 f"first_learn_iters_task={self.current_task}_globaliter={self.global_iters}.txt")
        np.savetxt(save_path, self.forget_stats["first_learn_iters"], delimiter=', ', fmt='%1.0f')

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
            self._log_forget_scores_hist(self.forget_scores)
            
            fs_dict = {"count prev_corrects": sum(self.forget_stats["prev_corrects"]),
                       "count corrects": sum(corrects),
                       "count never_correct": len(self.forget_stats["never_correct"]),
                       "count all_first_learnt": np.count_nonzero(np.isfinite(self.forget_stats["first_learn_iters"]))}

            wandb.log({k: v for k, v in fs_dict.items()}, step=self.global_iters)
        return

    def on_task_end(self):
        super(SupTrainerWForgetStats, self).on_task_end()
        self.save_forget_scores()
        self.save_first_learn_iters()
        return

    def _log_forget_scores_hist(self, fs, bins=20):
        if sum(np.isinf(fs)) > 0:
            fs[fs == np.inf] = -1
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(fs, bins=bins)
        plt.title(f"Forget scores at {self.global_iters} steps, task {self.current_task}")
        plt.xlabel("Number of forgetting events occurred")
        plt.ylabel("Number of training samples")
        wandb.log({"forget scores histogram": wandb.Image(fig)}, step=self.global_iters)
        plt.close()
        return


@gin.configurable(denylist=['device', 'model', 'data', 'logdir'])
class SupTrainerWReplay(SupTrainer):
    MEMORY_TYPES = ["fixed", "reservoir", "forgettables", "scorerank", "fixedscorerank", "fixedunforgettables"]

    def __init__(self, device, model, data, logdir, use_replay=gin.REQUIRED, memory_type=gin.REQUIRED,
                 replay_memory_size=None, replay_batch_size=None, precomputed_scores_path=None, score_type=None,
                 score_order=None, update_content_scores=None, check_containing=None, test_on_memcontent=False,
                 use_soft_forgets=False, softforget_pred_threshold=0.95, replace_newest=True):
        logging.info('Supervised trainer.')
        super().__init__(device, model, data, logdir)
        self.use_replay = use_replay

        if self.use_replay:
            logging.info('Use replay from memory.')

            err_message = "Replay memory type must be element of {}".format(self.MEMORY_TYPES)
            assert (memory_type in self.MEMORY_TYPES) == True, err_message

            err_message = "Parameter value must be set in config file"
            assert (replay_memory_size is not None) == True, err_message
            assert (replay_batch_size is not None) == True, err_message
            self.memory_type = memory_type
            self.replay_memory_size = replay_memory_size
            self.replay_batch_size = replay_batch_size
            self.precomputed_scores_path = precomputed_scores_path
            self.score_type = score_type
            self.score_order = score_order
            self.update_content_scores = update_content_scores
            self.check_containing = check_containing
            self.test_on_memcontent = test_on_memcontent
            self.use_soft_forgets = use_soft_forgets
            self.softforget_pred_threshold = softforget_pred_threshold
            self.replace_newest = replace_newest
            self.init_memory()

    def init_memory(self):
        if self.memory_type == "reservoir":
            self.replay_memory = memories.ReservoirMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size
            )
        elif self.memory_type == "fixed":
            self.replay_memory = memories.FixedMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size,
            )
        elif self.memory_type == "fixedscorerank":
            err_message = "Parameter value must be set in config file"
            assert (self.precomputed_scores_path is not None) == True, err_message
            assert (self.score_type  in ["forget", "consistency"]) == True, err_message
            assert (self.score_order in ["low", "high", "caws", "unforgettables"]) == True, err_message
            self.replay_memory = memories.FixedScoresRankMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size,
                precomputed_scores_path=self.precomputed_scores_path,
                dataset_indices_in_orig=[idx for idxl in self.data.train_task_datasets_indices_in_orig for idx in idxl],
                score_order=self.score_order,
                score_type=self.score_type,
            )
            if not os.path.isdir(os.path.join(self.logdir, "memory_content_scores")):
                os.makedirs(os.path.join(self.logdir, "memory_content_scores"))
        elif self.memory_type == "scorerank":
            err_message = "Parameter value must be set in config file"
            assert (self.precomputed_scores_path is not None) == True, err_message
            assert (self.score_type  in ["forget", "consistency"]) == True, err_message
            assert (self.score_order in ["low", "high"]) == True, err_message
            self.replay_memory = memories.PrecomputedScoresRankMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size,
                precomputed_scores_path=self.precomputed_scores_path,
                score_order=self.score_order,
            )
            if not os.path.isdir(os.path.join(self.logdir, "memory_content_scores")):
                os.makedirs(os.path.join(self.logdir, "memory_content_scores"))
        elif self.memory_type == "forgettables":
            err_message = "Parameter value must be set in config file"
            assert (self.score_order in ["low", "high", "best"]) == True, err_message
            assert (self.update_content_scores in [True, False]) == True, err_message
            assert (self.check_containing in [True, False]) == True, err_message
            self.replay_memory = memories.ForgettablesMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size,
                score_order=self.score_order,
                update_content_scores=self.update_content_scores,
                check_containing=self.check_containing,
                num_train_examples=len(self.data.train_dataset),
                logdir=self.logdir
            )
            if not os.path.isdir(os.path.join(self.logdir, "memory_content_forget_scores")):
                os.makedirs(os.path.join(self.logdir, "memory_content_forget_scores"))
        elif self.memory_type == "fixedunforgettables":
            err_message = "Parameter value must be set in config file"
            assert (self.score_order in ["low", "high", "unforgettables"]) == True, err_message
            assert (self.update_content_scores in [True, False]) == True, err_message
            assert (self.check_containing in [True, False]) == True, err_message
            self.replay_memory = memories.FixedUnforgettablesMemory(
                image_shape=self.data.input_shape,
                target_shape=(1,),
                device=self.device,
                size_limit=self.replay_memory_size,
                score_order=self.score_order,
                update_content_scores=self.update_content_scores,
                check_containing=self.check_containing,
                use_soft_forgets=self.use_soft_forgets,
                softforget_pred_threshold=self.softforget_pred_threshold,
                replace_newest=self.replace_newest,
                num_train_examples=len(self.data.train_dataset),
            )
            if not os.path.isdir(os.path.join(self.logdir, "memory_content_forget_scores")):
                os.makedirs(os.path.join(self.logdir, "memory_content_forget_scores"))

            if not os.path.isdir(os.path.join(self.logdir, "global_forget_scores")):
                os.makedirs(os.path.join(self.logdir, "global_forget_scores"))

            if not os.path.isdir(os.path.join(self.logdir, "first_learn_iters")):
                os.makedirs(os.path.join(self.logdir, "first_learn_iters"))

        if not os.path.isdir(os.path.join(self.logdir, "memory_content_idxinds")):
            os.makedirs(os.path.join(self.logdir, "memory_content_idxinds"))
        if not os.path.isdir(os.path.join(self.logdir, "memory_content_update_indices")):
            os.makedirs(os.path.join(self.logdir, "memory_content_update_indices"))
        return

    def calc_loss_on_batch(self, input_images, target):
        input_images_combined = input_images
        target_combined = target

        if self.use_replay and not self.replay_memory.empty():
            replay_images, replay_target = self.replay_memory.get_samples(self.replay_batch_size)
            input_images_combined = torch.cat([input_images, replay_images], dim=0)
            replay_target = replay_target.squeeze(dim=-1)
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
        corrects = torch.eq(predictions, target).detach().cpu().numpy().astype(int)
        accuracy_mean = torch.mean(torch.eq(predictions, target).float())

        softmax_output = torch.nn.functional.softmax(model_output, dim=-1).detach()
        softmax_preds = softmax_output.gather(1, target.unsqueeze(dim=1)).squeeze().cpu().numpy()

        results = {'total_loss_mean': loss_mean,
                   'accuracy': accuracy_mean,
                   'corrects': corrects,
                   'softmax_preds': softmax_preds}
        return results

    def _test_on_memcontent(self, mem_content):
        x_memcontent = (mem_content["images"], mem_content["targets"].flatten().type(torch.LongTensor))
        with torch.no_grad():
            self.model.eval()
            test_results = self.test_on_batch(x_memcontent)

            test_results = {key: value for key, value in test_results.items()
                            if torch.is_tensor(value) == True}

            test_results = {f'memcontent test_{key}': value for key, value in test_results.items()}
            template = ("Task {}/{}x{}\tTest on memcontent\tglobal iter: {} ({:.2f}%), metrics: "
                        + "".join([key + ": {:.3f}  " for key in test_results.keys()]))

            logging.info(template.format(self.current_task,
                                        self.num_tasks,
                                        self.num_cycles,
                                        self.global_iters,
                                        float(self.global_iters) / self.iters * 100.,
                                        *[item.data for item in test_results.values()]))

            for metric, result in test_results.items():
                wandb.log({metric: result}, step=self.global_iters)
        return

    def _log_replay_memory_images(self):
        replay_images, replay_target = self.replay_memory.get_samples(max(self.replay_memory_size, 500))
        replay_images = self.data.inverse_normalize(replay_images.detach().cpu())
        fig = plt.figure(figsize = (20, 40))
        image_grid = torchvision.utils.make_grid(replay_images)
        ax = plt.imshow(np.transpose(image_grid, (1, 2, 0)))
        plt.axis('off')
        fig = plt.gcf()
        wandb.log({"replay memory content": fig}, step=self.global_iters)
        plt.close()
        return

    def _log_replay_memory_class_distribution(self):
        fig = plt.figure(figsize = (10, 5))
        classes = list(self.replay_memory.target2indices.keys())
        counts = [len(indices) for indices in self.replay_memory.target2indices.values()]
        plt.bar(classes, counts, color ='maroon', width=0.2)
        plt.xlabel("Classes in memory")
        plt.ylabel("Number of images")
        plt.title("Class distribution of images in replay memory")
        wandb.log({"replay memory class distribution": wandb.Image(fig)}, step=self.global_iters)
        plt.close()
        return

    def on_iter_end(self, batch, batch_results):
        is_task_start_or_end_iter = self.iter_count < 10 or self.iter_count > self.iters_per_task - 10
        if (self.global_iters % self.log_interval == 0) or is_task_start_or_end_iter:
            wandb.log({"count memory content update": self.replay_memory.count_content_update}, step=self.global_iters)
            if not self.replay_memory.empty():
                self._log_replay_memory_images()
                self._log_replay_memory_class_distribution()
                if self.test_on_memcontent:
                    self._test_on_memcontent(self.replay_memory.content)
            else:
                logging.info("Replay memory is currently empty.")
            if self.memory_type == "scorerank" or self.memory_type == "fixedscorerank":
                self._log_scores_hist(self.replay_memory.precomputed_scores,
                                      f"precomputed {self.score_type} scores histogram",
                                      score_type=self.score_type)
                self._log_scores_hist(self.replay_memory.content["scores"],
                                      f"memory content precomputed {self.score_type} scores histogram",
                                      score_type=self.score_type)
                fs_dict = {"memory content size": len(self.replay_memory.content["indices_in_ds"]),
                           "memory content score min": min(self.replay_memory.content["scores"]),
                           "memory content score max": max(self.replay_memory.content["scores"]),
                           "global score min": min(self.replay_memory.precomputed_scores),
                           "global score max": max(self.replay_memory.precomputed_scores)}
                wandb.log({k: v for k, v in fs_dict.items()}, step=self.global_iters)

            elif self.memory_type == "forgettables" or self.memory_type == "fixedunforgettables":
                self._log_scores_hist(self.replay_memory.global_forget_scores,
                                      "forget scores histogram")

                self._log_scores_hist(self.replay_memory.content["forget_scores"],
                                      "memory content forget scores histogram")

                self._log_scores_hist(self.replay_memory.forget_stats["num_softforgets"],
                                      "soft forget scores histogram")

                fs_dict = {"count prev_corrects": sum(self.replay_memory.forget_stats["prev_corrects"]),
                           "count corrects": sum(batch_results["corrects"]),
                           "count never_correct": len(self.replay_memory.forget_stats["never_correct"]),
                           "count all_first_learnt": np.count_nonzero(np.isfinite(self.replay_memory.forget_stats["first_learn_iters"])),
                           "memory content size": self.replay_memory.size,
                           "memory content score min": min(self.replay_memory.content["forget_scores"]),
                           "memory content score max": max(self.replay_memory.content["forget_scores"]),
                           "global score min": min(self.replay_memory.global_forget_scores),
                           "global score max": max(self.replay_memory.global_forget_scores)}
                wandb.log({k: v for k, v in fs_dict.items()}, step=self.global_iters)
            if is_task_start_or_end_iter and not self.replay_memory.empty():
                save_path = os.path.join(self.logdir,
                                        "memory_content_idxinds",
                                        f"memory_idxinds_task={self.current_task}_globaliter={self.global_iters}.txt")
                existing_indices = np.array([i for i in self.replay_memory.content["indices_in_ds"] if i is not None])
                np.savetxt(save_path, existing_indices, delimiter=', ', fmt='%1.0f')
                save_path = os.path.join(self.logdir,
                                        "memory_content_update_indices",
                                        f"memory_content_update_indices_task={self.current_task}_globaliter={self.global_iters}.txt")
                np.savetxt(save_path, np.array(self.replay_memory.content_update_indices), delimiter=', ', fmt='%1.0f')

                if self.memory_type == "scorerank" or self.memory_type == "fixedscorerank":
                    save_path = os.path.join(self.logdir,
                                            "memory_content_scores",
                                            f"memory_scores_task={self.current_task}_globaliter={self.global_iters}.txt")
                    np.savetxt(save_path, self.replay_memory.content["scores"], delimiter=', ', fmt='%1.3f')

                elif self.memory_type == "forgettables" or self.memory_type == "fixedunforgettables":
                    save_path = os.path.join(self.logdir,
                                            "memory_content_forget_scores",
                                            f"memory_fs_task={self.current_task}_globaliter={self.global_iters}.txt")
                    np.savetxt(save_path, self.replay_memory.content["forget_scores"], delimiter=', ', fmt='%1.0f')
                    save_path = os.path.join(self.logdir,
                                            "global_forget_scores",
                                            f"global_fs_task={self.current_task}_globaliter={self.global_iters}.npy")
                    np.save(save_path, self.replay_memory.global_forget_scores)
                if self.memory_type == "fixedunforgettables":
                    save_path = os.path.join(self.logdir,
                                            "first_learn_iters",
                                            f"first_learn_iters_task={self.current_task}_globaliter={self.global_iters}.txt")
                    np.savetxt(save_path, self.replay_memory.forget_stats["first_learn_iters"], delimiter=', ', fmt='%1.0f')

        if self.use_replay:
            self.replay_memory.on_batch_end(*batch, batch_results["corrects"], batch_results["softmax_preds"], self.global_iters)
            if self.memory_type == "fixedunforgettables":
                count_first_learns = len(np.where(self.replay_memory.forget_stats["first_learn_iters"] == self.global_iters)[0])
                wandb.log({"count first_learns in iter": count_first_learns}, step=self.global_iters)
                fs_dict = {"forgetstats/num forgets in iter": self.replay_memory.num_forgets_in_iter,
                           "forgetstats/num softforgets in iter": self.replay_memory.num_softforgets_in_iter,
                           "forgetstats/num common indices forgetting_and_softforgetting": self.replay_memory.num_common_forgetting_and_softforgetting}
                wandb.log({k: v for k, v in fs_dict.items()}, step=self.global_iters)
        super(SupTrainerWReplay, self).on_iter_end(batch, batch_results)
        return

    def _log_scores_hist(self, scores_input, log_name, score_type="forget", bins=20):
        scores = scores_input.copy()
        if sum(np.isinf(scores)) > 0:
            scores[scores == np.inf] = -1
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(scores, bins=bins)
        plt.title(f"{score_type.capitalize()} scores at {self.global_iters} steps, task {self.current_task}")
        plt.xlabel(f"{score_type.capitalize()} score values")
        plt.ylabel("Number of training examples")
        wandb.log({log_name: wandb.Image(fig)}, step=self.global_iters)
        plt.close()
        return
