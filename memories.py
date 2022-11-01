from abc import abstractmethod
import numpy as np
import os
import random
import torch


class Memory:
    def __init__(self, image_shape, target_shape, device, size_limit):
        self.size_limit = size_limit
        self.size = 0
        self.target2indices = {}
        self.content = {
            'images': torch.zeros((self.size_limit, *image_shape), device=device),
            'targets': torch.zeros((self.size_limit, *target_shape), dtype=torch.int32, device=device)
        }

    @abstractmethod
    def _update_with_item(self, update_image, updat_target):
        return

    def on_batch_end(self, update_images, update_targets, *args):
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


class ForgettablesMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit, size_limit_per_target, num_train_examples,
                 logdir, log_score_freq=100):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.size_limit_per_target = size_limit_per_target
        self.size_per_target = {}
        self.num_train_examples = num_train_examples
        self.logdir = logdir

        self.forget_stats = {
            "prev_corrects": np.zeros(self.num_train_examples, dtype=np.int32),
            "num_forgets": np.zeros(self.num_train_examples, dtype=float),
            "never_correct": np.arange(self.num_train_examples, dtype=np.int32),
        }

        self.global_forget_scores = self.forget_stats["num_forgets"].copy()
        self.global_forget_scores[self.forget_stats["never_correct"]] = np.inf
        self.log_score_freq = log_score_freq
        if not os.path.isdir(os.path.join(self.logdir, "global_forget_scores")):
            os.makedirs(os.path.join(self.logdir, "global_forget_scores"))
        self.content.update({"forget_scores": np.zeros(self.size_limit, dtype=float)})

    def _update_forget_stats(self, idxs, corrects):
        idxs_where_forgetting = idxs[self.forget_stats["prev_corrects"][idxs] > corrects]
        self.forget_stats["num_forgets"][idxs_where_forgetting] += 1
        self.forget_stats["prev_corrects"][idxs] = corrects
        self.forget_stats["never_correct"] = np.setdiff1d(
            self.forget_stats["never_correct"],
            idxs[corrects.astype(bool)],
            True
        )
        self.global_forget_scores = self.forget_stats["num_forgets"].copy()
        self.global_forget_scores[self.forget_stats["never_correct"]] = np.inf
        return

    def _save_forget_scores(self, global_iters):
        save_path = os.path.join(self.logdir,
                                 "global_forget_scores",
                                 f"fs_globaliter={global_iters}.npy")
        np.save(save_path, self.global_forget_scores)

    def _remove_idx_with_target(self, idx, target):
        old_target = self.content['targets'][idx].item()
        self.target2indices[old_target].remove(idx)

    def _update_content_at_idx(self, update_image, update_target, idx, forget_score):
        super(ForgettablesMemory, self)._update_content_at_idx(update_image, update_target, idx)
        self.content["forget_scores"][idx] = forget_score

    def _update_with_item(self, update_image, update_target, update_idx_in_ds):
        min_forget_score = self.content["forget_scores"].min()
        update_forget_score = self.global_forget_scores[update_idx_in_ds]
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0

        if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
            idx = self.size
            self._update_content_at_idx(update_image, update_target, idx, update_forget_score)
            self.size += 1
            self.size_per_target[target_value] += 1

        elif update_forget_score < min_forget_score:
            idx = np.argmin(self.content["forget_scores"])
            self._remove_idx_with_target(idx, update_target)
            self._update_content_at_idx(update_image, update_target, idx, update_forget_score)
            self.size += 1
        return

    def on_batch_end(self, update_images, update_targets, indices_in_ds, corrects, global_iters):
        self._update_forget_stats(indices_in_ds, corrects)
        self.min_forget_score = self.content["forget_scores"].min()
        for i in range(update_images.shape[0]):
            self._update_with_item(update_images[i], update_targets[i], indices_in_ds[i])

        if global_iters % self.log_score_freq == 0:
            self._save_forget_scores(global_iters)
        return