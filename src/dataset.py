import torch
import random
import numpy as np
from torchvision import datasets
from torchvision import transforms


class CustomMNISTDataset(datasets.MNIST):
    def __init__(self, cfg, train, download=False):
        self.imbalance_r = cfg.DATA_SET.imbalance
        self.noise_r = cfg.DATA_SET.noise

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((cfg.DATA_SET.mean,), (cfg.DATA_SET.std,))
        ])
        super(CustomMNISTDataset, self).__init__(cfg.DATA_SET.root, train, transform, None, download)

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
