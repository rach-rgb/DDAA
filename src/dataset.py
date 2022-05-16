import logging

import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from utils import step_to_tensor


class ToFloatTensor(object):
    def __call__(self, img):
        return torch.unsqueeze(img.type(torch.FloatTensor) / 255, 0)


tr_MNIST = transforms.Compose([
    ToFloatTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

tr_CIFAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


# create messy dataset
class RawDataset(data.Dataset):
    def __init__(self, cfg, dataset, mess, index=None, transform=None):
        x = dataset.data
        y = dataset.targets
        if isinstance(y, list):
            y = torch.tensor(y)

        # get subset
        if index is not None:
            x, y = self.get_subset(index, x, y)

        if mess:  # apply mess ratio for train set
            x, y = self.make_mess(cfg, x, y)

        # class distribution
        self.n_classes = cfg.DATA_SET.num_classes
        self.n_per_classes = [len(np.where(y == cls)[0]) for cls in range(0, self.n_classes)]

        self.data = x  # Tensor [num_of_data, (C), H, W]
        self.targets = y  # Tensor [num_of_data]

        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_subset(self, index, x, y):
        mask = np.zeros(len(y), dtype=bool)
        mask[index] = True

        return x[mask], y[mask]

    def make_mess(self, cfg, x, y):
        imbalance_r = cfg.DATA_SET.imbalance
        noise_r = cfg.DATA_SET.noise
        num_classes = cfg.DATA_SET.num_classes
        assert 0 <= imbalance_r <= 1
        assert 0 <= noise_r <= 1

        # apply class imbalance
        if imbalance_r != 1:
            logging.info("Apply class imbalance ratio: %0.3f", imbalance_r)
            small_label = [x for x in range(0, int(num_classes / 2))]

            remove_idx = []
            ny = y.numpy()
            for label in small_label:
                label_idx = np.where(ny == label)[0]
                remove_idx = remove_idx + list(label_idx[:int(len(label_idx) * (1-imbalance_r))])

            mask = np.ones(len(y), dtype=bool)
            mask[remove_idx] = False

            x = x[mask]
            y = y[mask]

        # apply label noise
        if noise_r != 0:
            logging.info("Apply noise ratio: %0.2f", noise_r)
            noise = torch.randint(num_classes - 1, size=(int(len(y) * noise_r),))
            y = torch.cat([y[:len(y) - int(len(y) * noise_r)], noise])

        return x, y


# Create Dataset from steps
class StepDataset(data.Dataset):
    def __init__(self, n_classes, steps):
        self.data, self.targets = step_to_tensor(steps)
        self.data = torch.stack(self.data, dim=0)  # Tensor [num_of_data, C, W, H]
        self.targets = torch.tensor(self.targets)  # Tensor [num_of_data]

        self.transform = transforms.Compose([])  # empty

        # class distribution
        self.n_classes = n_classes
        self.n_per_classes = [int(len(self.data) / n_classes)] * n_classes

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
