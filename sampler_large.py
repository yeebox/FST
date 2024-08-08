import random
from torch.utils.data import Sampler, DataLoader
# from tiny_imagenet import *
from torchvision import datasets, transforms
import torch
import numpy as np


class CustomSampler(Sampler):
    def __init__(self, data_source, batch_size, num_classes):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_indices = self._create_class_indices()

        self.reset()

    def reset(self):
        self.remaining_indices = set(range(len(self.data_source)))

    def _create_class_indices(self):
        class_indices = {}
        for idx, label in enumerate(self.data_source.targets):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __iter__(self):
        self.reset()
        cnt_cls = 0
        while self.remaining_indices:
            if cnt_cls  % len(list(self.class_indices.keys())) == 0:
                rand_list = list(self.class_indices.keys()).copy()
                random.shuffle(rand_list)
                i = 0
            selected_indices = []
            # ========================= the second choice for sampling =================================
            # selected_classes = np.random.choice(list(self.class_indices.keys()), size=self.num_classes, replace=False)
            # ========================= the second choice for sampling =================================
            selected_classes = rand_list[i: i + self.num_classes]
            selected_classes = np.array(selected_classes)
            i = i + self.num_classes
            cnt_cls = cnt_cls + self.num_classes

            cnt = 0
            for class_idx in selected_classes:
                temp_idx = list(set(self.class_indices[class_idx]) & self.remaining_indices)
                if len(temp_idx) < self.batch_size // self.num_classes:
                    continue
                else:
                    cnt += self.batch_size // self.num_classes
            if cnt == self.batch_size:
                for class_idx in selected_classes:
                    available_indices = list(set(self.class_indices[class_idx]) & self.remaining_indices)
                    selected_indices.extend(
                        np.random.choice(available_indices, size=self.batch_size // self.num_classes, replace=False))
            elif len(self.remaining_indices) >= self.batch_size:
                selected_indices.extend(np.random.choice(list(self.remaining_indices), size=self.batch_size, replace=False))
            else:
                selected_indices.extend(np.random.choice(list(self.remaining_indices), size=len(self.remaining_indices), replace=False))

            random.shuffle(selected_indices)
            self.remaining_indices -= set(selected_indices)
            yield selected_indices

    def __len__(self):
        return len(self.data_source) // self.batch_size



