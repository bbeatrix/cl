from abc import abstractmethod
import logging
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
            'targets': torch.zeros((self.size_limit, *target_shape), dtype=torch.int32, device=device),
            'indices_in_ds': [None] * self.size_limit,
        }
        self.count_content_update = 0
        self.content_update_indices = []

    @abstractmethod
    def _update_with_item(self, update_image, updat_target, update_index_in_ds):
        return

    def on_batch_end(self, update_images, update_targets, indices_in_ds, *args):
        for i in range(update_images.shape[0]):
            self._update_with_item(update_images[i], update_targets[i], indices_in_ds[i])

    def _update_content_at_idx(self, update_image, update_target, index_in_ds, idx):
        self.content['images'][idx] = update_image
        self.content['targets'][idx] = update_target
        self.content['indices_in_ds'][idx] = index_in_ds

        update_target_value = update_target.item()
        if update_target_value in self.target2indices.keys():
            self.target2indices[update_target_value].append(idx)
        else:
            self.target2indices[update_target_value] = [idx]
        self.count_content_update += 1
        self.content_update_indices.append(idx)

    def get_samples(self, sample_size, target=None):
        if target is None:
            sample_size = min(self.size, sample_size)
            selected_indices = np.random.choice(range(self.size), sample_size, replace=False)
        else:
            indices = self.target2indices[target]
            sample_size = min(sample_size, len(indices))
            selected_indices = np.random.choice(indices, sample_size, replace=False)

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

    def _update_with_item(self, update_image, update_target, update_index_in_ds):
        if self.size < self.size_limit:
            idx = self.size
            self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
            self.size += 1
        else:
            # memory is full.
            m = random.randrange(self.num_seen_images_in_stream)
            if m < self.size_limit:
                # Put it in
                idx = m
                self._remove_idx_with_target(idx, update_target)
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
        self.num_seen_images_in_stream += 1


class FixedMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.size_per_target = {}
        self.size_limit_per_target = self.size_limit

    def get_index_of_replace(self):
        for target, size in self.size_per_target.items():
            if size > self.size_limit_per_target:
                idx = self.target2indices[target][0]
                self.target2indices[target].pop(0)
                self.size_per_target[target] -= 1
                return idx
        return

    def _update_with_item(self, update_image, update_target, update_index_in_ds):
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
            self.size_limit_per_target = self.size_limit // len(self.size_per_target)
        if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
            if update_index_in_ds not in self.content["indices_in_ds"]:
                idx = self.size
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
                self.size += 1
                self.size_per_target[target_value] += 1
        elif self.size_per_target[target_value] < self.size_limit_per_target:
            if update_index_in_ds not in self.content["indices_in_ds"]:
                idx = self.get_index_of_replace()
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
                self.size_per_target[target_value] += 1


class PrecomputedScoresRankMemory(FixedMemory):
    def __init__(self, image_shape, target_shape, device, size_limit, precomputed_scores_path, score_order):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.precomputed_scores = np.load(precomputed_scores_path)
        self.score_order = score_order
        self.content.update({"scores": -np.ones(self.size_limit, dtype=float)})

    def _update_content_at_idx(self, update_image, update_target, update_index_in_ds, idx):
        super(PrecomputedScoresRankMemory, self)._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
        self.content["scores"][idx] = self.precomputed_scores[update_index_in_ds]

    def get_index_of_replace(self):
        for target, size in self.size_per_target.items():
            if size > self.size_limit_per_target:
                scores = self.content["scores"][self.target2indices[target]]
                if self.score_order == "low":
                    replace_score_idx = np.argmax(scores)
                else:
                    replace_score_idx = np.argmin(scores)
                idx = self.target2indices[target][replace_score_idx]
                self.target2indices[target].pop(replace_score_idx)
                self.size_per_target[target] -= 1
                return idx
        return

    def _update_with_item(self, update_image, update_target, update_index_in_ds):
        super()._update_with_item(update_image, update_target, update_index_in_ds)
        
        if update_index_in_ds not in self.content["indices_in_ds"]:
            scores_in_content = self.content["scores"][self.target2indices[update_target.item()]]
            replace_idx_in_content = None
            if self.score_order == "low" and np.max(scores_in_content) > self.precomputed_scores[update_index_in_ds]:
                replace_idx_in_content = np.argmax(scores_in_content)
            elif np.min(scores_in_content) < self.precomputed_scores[update_index_in_ds]:
                replace_idx_in_content = np.argmin(scores_in_content)
            if replace_idx_in_content is not None:
                self.target2indices[update_target.item()].pop(replace_idx_in_content)
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, replace_idx_in_content)
        return


class FixedScoresRankMemory(FixedMemory):
    def __init__(self, image_shape, target_shape, device, size_limit, precomputed_scores_path, score_order, score_type):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.precomputed_scores_with_labels = np.load(precomputed_scores_path)
        self.precomputed_scores = self.precomputed_scores_with_labels[0]
        self.score_type = score_type
        self.score_order = score_order
        self.content.update({"scores": -100 * np.ones(self.size_limit, dtype=float)})
        self.selected_indices_per_class = {}

    def _update_content_at_idx(self, update_image, update_target, update_index_in_ds, idx):
        super(FixedScoresRankMemory, self)._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
        self.content["scores"][idx] = self.precomputed_scores[update_index_in_ds]

    def _select_indices_per_target(self, target):
        class_indices = np.where(self.precomputed_scores_with_labels[1] == target)[0]
        class_scores = self.precomputed_scores_with_labels[0][class_indices]
        # when encountering the first item we do not want to populate the whole memory for sure
        selection_size = min(self.size_limit//2, self.size_limit_per_target) 

        if self.score_type == "consistency":
            if self.score_order == "low":
                sorted_class_score_indices = np.argsort(class_scores)
                selected_indices = sorted_class_score_indices[:selection_size]
            elif self.score_order == "high":
                sorted_class_score_indices = np.argsort(class_scores)
                selected_indices = np.flip(sorted_class_score_indices)[:selection_size]
            elif self.score_order == "caws":
                selected_per_class = np.where(class_scores > 0.6)[0]
                selection_size=min(selection_size, len(selected_per_class))
                selected_indices = np.random.choice(selected_per_class, selection_size, replace=False)

        elif self.score_type == "forget":
            if self.score_order == "low":
                forgettables_indices = np.where(class_scores > 0)[0]
                sorted_forgettables_indices = np.argsort(class_scores[forgettables_indices])
                selected_indices = forgettables_indices[sorted_forgettables_indices[:selection_size]]
            elif self.score_order == "high":
                forgettables_indices = np.where(class_scores > 0)[0]
                sorted_forgettables_indices = np.argsort(class_scores[forgettables_indices])
                selected_indices = forgettables_indices[np.flip(sorted_forgettables_indices)[:selection_size]]
            elif self.score_order == "unforgettables":
                unforgettables_indices = np.where(class_scores == 0)[0]
                selected_indices = unforgettables_indices[:selection_size]

        self.selected_indices_per_class[target] = list(class_indices[selected_indices])
        return

    def get_index_of_replace(self):
        for target, size in self.size_per_target.items():
            if size > self.size_limit_per_target:
                idx_in_ds_to_remove = self.selected_indices_per_class[target][-1]
                l = len(self.selected_indices_per_class[target])
                while idx_in_ds_to_remove not in self.content["indices_in_ds"]:
                    l -= 1
                    logging.info(f"Removing excess index {idx_in_ds_to_remove} from selected_indices_per_class[{target}][{l}]")
                    idx_in_ds_to_remove = self.selected_indices_per_class[target].pop(-1)
                idx = np.where(self.content["indices_in_ds"] == idx_in_ds_to_remove)[0][0]
                self.target2indices[target].remove(idx)
                self.size_per_target[target] -= 1
                return idx
        return

    def _update_with_item(self, update_image, update_target, update_index_in_ds):
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
            self.size_limit_per_target = self.size_limit // len(self.size_per_target)
            self._select_indices_per_target(target_value)
            logging.info(f"Added new target {target_value}, new size_limit_per_target is {self.size_limit_per_target}")
        if update_index_in_ds.item() in self.selected_indices_per_class[target_value] and update_index_in_ds.item() not in self.content["indices_in_ds"]:
            if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.size
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
                self.size += 1
                self.size_per_target[target_value] += 1
            elif self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.get_index_of_replace()
                self._update_content_at_idx(update_image, update_target, update_index_in_ds, idx)
                self.size_per_target[target_value] += 1


class FixedUnforgettablesMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit, score_order, update_content_scores, check_containing,
                 num_train_examples):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.score_order = score_order
        self.update_content_scores = update_content_scores
        self.check_containing = check_containing
        self.size_limit_per_target = size_limit
        self.size_per_target = {}
        self.num_train_examples = num_train_examples

        self.forget_stats = {
            "prev_corrects": np.zeros(self.num_train_examples, dtype=np.int32),
            "num_forgets": np.zeros(self.num_train_examples, dtype=float),
            "never_correct": np.arange(self.num_train_examples, dtype=np.int32),
            "first_learn_iters": np.inf * np.ones(self.num_train_examples, dtype=np.int32),
        }

        self.global_forget_scores = self.forget_stats["num_forgets"].copy()
        self.global_forget_scores[self.forget_stats["never_correct"]] = np.inf

        self.content.update({"forget_scores": np.inf * np.ones(self.size_limit, dtype=float)})

    def _update_forget_stats(self, idxs, corrects, global_iters):
        for i, idx in enumerate(idxs):
            if self.forget_stats["first_learn_iters"][idx] == np.inf and corrects[i] == 1:
                self.forget_stats["first_learn_iters"][idx] = global_iters
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

    def _remove_idx_with_target(self, idx, target):
        old_target = self.content['targets'][idx].item()
        self.target2indices[old_target].remove(idx)

    def _update_content_at_idx(self, update_image, update_target, update_idx_in_ds, idx, forget_score):
        super(FixedUnforgettablesMemory, self)._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx)
        self.content["forget_scores"][idx] = forget_score

    def get_index_of_replace(self):
        for target, size in self.size_per_target.items():
            if size > self.size_limit_per_target:
                scores = self.content["forget_scores"][self.target2indices[target]]
                if self.score_order == "low":
                    replace_score_idx = np.argwhere(scores == np.max(scores)).flatten()[-1]
                else:
                    replace_score_idx = np.argwhere(scores == np.min(scores)).flatten()[-1]
                idx = self.target2indices[target][replace_score_idx]
                self.target2indices[target].pop(replace_score_idx)
                self.size_per_target[target] -= 1
                return idx
        return

    def _update_with_item(self, update_image, update_target, update_idx_in_ds):
        min_forget_score = self.content["forget_scores"].min()
        update_forget_score = self.global_forget_scores[update_idx_in_ds]
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
            self.size_limit_per_target = self.size_limit // len(self.size_per_target)
        if self.check_containing == False or (self.check_containing == True and update_idx_in_ds not in self.content["indices_in_ds"]):
            if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.size
                self._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx, update_forget_score)
                self.size += 1
                self.size_per_target[target_value] += 1

            elif self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.get_index_of_replace()
                self._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx, update_forget_score)
                self.size_per_target[target_value] += 1

            else:
                scores_in_content = self.content["forget_scores"][self.target2indices[target_value]]
                replace_idx_in_content = None
                unforgettables_conditions = (self.score_order == "unforgettables" and np.max(scores_in_content) > update_forget_score == 0)
                if (self.score_order == "low" and np.max(scores_in_content) > update_forget_score > 0) or (unforgettables_conditions):
                    replace_score_idx = np.argmax(scores_in_content)
                    replace_idx_in_content = self.target2indices[target_value][replace_score_idx]
                    self.target2indices[target_value].pop(replace_score_idx)
                elif self.score_order == "high" and np.min(scores_in_content) < update_forget_score > 0:
                    replace_score_idx = np.argmin(scores_in_content)
                    replace_idx_in_content = self.target2indices[target_value][replace_score_idx]
                    self.target2indices[target_value].pop(replace_score_idx)
                if replace_idx_in_content is not None:
                    self._update_content_at_idx(update_image, update_target, update_idx_in_ds, replace_idx_in_content, update_forget_score)
        return

    def on_batch_end(self, update_images, update_targets, indices_in_ds, corrects, global_iters):
        self._update_forget_stats(indices_in_ds, corrects, global_iters)
        if self.update_content_scores:
            indices_where_content = np.where(np.not_equal(self.content["indices_in_ds"], None))[0]
            content_indices_in_ds = [self.content["indices_in_ds"][i] for i in indices_where_content]
            self.content["forget_scores"][indices_where_content] = self.global_forget_scores[content_indices_in_ds]
        for i in range(update_images.shape[0]):
            self._update_with_item(update_images[i], update_targets[i], indices_in_ds[i])

        return


class ForgettablesMemory(Memory):
    def __init__(self, image_shape, target_shape, device, size_limit, score_order, update_content_scores, check_containing,
                 num_train_examples, logdir, log_score_freq=100):
        super().__init__(image_shape, target_shape, device, size_limit)

        self.score_order = score_order
        self.update_content_scores = update_content_scores
        self.check_containing = check_containing
        self.size_limit_per_target = size_limit
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

    def _update_content_at_idx(self, update_image, update_target, update_idx_in_ds, idx, forget_score):
        super(ForgettablesMemory, self)._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx)
        self.content["forget_scores"][idx] = forget_score

    def get_index_of_replace(self):
        for target, size in self.size_per_target.items():
            if size > self.size_limit_per_target:
                scores = self.content["forget_scores"][self.target2indices[target]]
                if self.score_order == "low":
                    replace_score_idx = np.argmax(scores)
                else:
                    replace_score_idx = np.argmin(scores)
                idx = self.target2indices[target][replace_score_idx]
                self.target2indices[target].pop(replace_score_idx)
                self.size_per_target[target] -= 1
                return idx
        return

    def _update_with_item(self, update_image, update_target, update_idx_in_ds):
        min_forget_score = self.content["forget_scores"].min()
        update_forget_score = self.global_forget_scores[update_idx_in_ds]
        target_value = update_target.item()
        if target_value not in self.size_per_target.keys():
            self.size_per_target[target_value] = 0
            self.size_limit_per_target = self.size_limit // len(self.size_per_target)
        if self.check_containing == False or (self.check_containing == True and update_idx_in_ds not in self.content["indices_in_ds"]):
            if self.size < self.size_limit and self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.size
                self._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx, update_forget_score)
                self.size += 1
                self.size_per_target[target_value] += 1

            elif self.size_per_target[target_value] < self.size_limit_per_target:
                idx = self.get_index_of_replace()
                self._update_content_at_idx(update_image, update_target, update_idx_in_ds, idx, update_forget_score)
                self.size_per_target[target_value] += 1

            else:
                scores_in_content = self.content["forget_scores"][self.target2indices[update_target.item()]]
                replace_idx_in_content = None
                if self.score_order == "low" and np.max(scores_in_content) > update_forget_score:
                    replace_idx_in_content = np.argmax(scores_in_content)
                    self.target2indices[update_target.item()].pop(replace_idx_in_content)
                elif self.score_order == "best" and update_forget_score < min_forget_score: 
                    replace_idx_in_content = np.argmin(self.content["forget_scores"])
                    self._remove_idx_with_target(replace_idx_in_content, update_target)
                elif self.score_order == "high" and np.min(scores_in_content) < update_forget_score:
                    replace_idx_in_content = np.argmin(scores_in_content)
                    self.target2indices[update_target.item()].pop(replace_idx_in_content)
                if replace_idx_in_content is not None:
                    self._update_content_at_idx(update_image, update_target, update_idx_in_ds, replace_idx_in_content, update_forget_score)
        return

    def on_batch_end(self, update_images, update_targets, indices_in_ds, corrects, global_iters):
        self._update_forget_stats(indices_in_ds, corrects)
        if self.update_content_scores:
            indices_where_content = np.where(np.not_equal(self.content["indices_in_ds"], None))
            self.content["forget_scores"][indices_where_content] = self.global_forget_scores[indices_where_content]
        for i in range(update_images.shape[0]):
            self._update_with_item(update_images[i], update_targets[i], indices_in_ds[i])

        if global_iters % self.log_score_freq == 0:
            self._save_forget_scores(global_iters)
        return


def test():
    IMAGE_SHAPE=(1,)
    TARGET_SHAPE=(1,)
    device="cuda"
    memory = FixedUnforgettablesMemory(
        image_shape=IMAGE_SHAPE,
        target_shape=TARGET_SHAPE,
        device=device,
        size_limit=100,
        score_order='high',
        update_content_scores=True,
        check_containing=True,
        num_train_examples=50000)

    BATCH_SIZE=50
    DATASET_SIZE=50000
    xs = torch.tensor(np.random.rand(*((DATASET_SIZE,) + IMAGE_SHAPE)))
    ys = torch.tensor(np.zeros((DATASET_SIZE,)+ IMAGE_SHAPE, dtype=np.int32))
    ys[DATASET_SIZE//2:] = torch.tensor(np.ones((DATASET_SIZE//2, *IMAGE_SHAPE)))

    for i in range(1000):
        ii = i % (DATASET_SIZE // BATCH_SIZE)
        xb = xs[ii*BATCH_SIZE:(ii+1)*BATCH_SIZE]
        yb = ys[ii*BATCH_SIZE:(ii+1)*BATCH_SIZE]

        if i == 999:
            memory.on_batch_end(xb, yb, np.arange(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE), np.zeros((BATCH_SIZE,)), i)
        else:
            memory.on_batch_end(xb, yb, np.arange(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE), np.ones((BATCH_SIZE,)), i)

    print(memory.content)

if __name__ == "__main__":
    test()
