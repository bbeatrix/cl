import numpy as np
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
