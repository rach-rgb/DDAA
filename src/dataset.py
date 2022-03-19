from torchvision import datasets
import torch
from typing import Callable, Optional
import random
import numpy as np
from torchvision import transforms


class CustomMNISTDataset(datasets.MNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            imbalance_r=1,
            noise_r=0
    ):
        self.imbalance_r = imbalance_r
        self.noise_r = noise_r

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        super(CustomMNISTDataset, self).__init__(root, train, transform, target_transform, download)

    def _load_data(self):
        data, targets = super(CustomMNISTDataset, self)._load_data()

        if self.train is True:
            # apply class imbalance
            if self.imbalance_r != 1:
                small_label = random.sample(range(9), 5)
                small_size = int(len(data) / len(self.classes) * self.imbalance_r)

                remove_idx = []
                nx = targets.numpy()
                for label in small_label:
                    remove_idx = remove_idx + list(np.where(nx == label)[0][:small_size])

                mask = np.ones(len(data), dtype=bool)
                mask[remove_idx] = False

                data = data[mask]
                targets = targets[mask]

            # apply label noise
            if self.noise_r != 0:
                noise = torch.randint(9, size=(int(len(data) * self.noise_r),))
                targets = torch.cat([targets[:len(targets) - int(len(targets) * self.noise_r)], noise])

        return data, targets
