import os
import logging
from pathlib import Path

import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from src.utils import step_to_tensor
import src.custom_dataset.transform as tr


# Terminal
# import os, sys, logging
# from pathlib import Path
#
# import torch
# import numpy as np
# import torch.utils.data as data
# from torchvision import transforms, datasets
# from sklearn.model_selection import train_test_split
#
# sys.path.insert(0, '../src')
# import custom_dataset.transform as tr
# from utils import step_to_tensor

# return dataset
# Args: cfg(Config)
def get_dataset(cfg):
    is_MNIST = cfg.DATA_SET.name == 'MNIST'

    if is_MNIST:
        total_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        tr_train = tr.train_MNIST
        tr_valid = tr.test_MNIST
    else:   # CIFAR-10
        total_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=True, download=True)
        tr_train = tr.train_CIFAR
        tr_valid = tr.test_CIFAR

    if cfg.DATA_SET.source == 'load':
        output_dir = os.path.join(Path(os.getcwd()).parent, 'output', 'augment-' + cfg.DATA_SET.name)

        train_dataset = AugDataset(output_dir)
        val_dataset = None
        logging.info("Train dataset loaded from %s", output_dir)
    elif cfg.DATA_SET.train_split:  # source == Raw
        train_idx, val_idx, _, _ = train_test_split(range(len(total_dataset)), total_dataset.targets,
                                                    stratify=total_dataset.targets, test_size=cfg.DATA_SET.val_size)
        train_dataset = RawDataset(cfg, total_dataset, mess=True, index=train_idx, transform=tr_train)
        val_dataset = RawDataset(cfg, total_dataset, mess=False, index=val_idx, transform=tr_valid)
    else:
        train_dataset = RawDataset(cfg, total_dataset, mess=True, transform=tr_train)
        val_dataset = None

    return train_dataset, val_dataset


# create custom dataset
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

        self.do_transform = True
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        if self.transform is not None and self.do_transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    # add augmentor module
    # Args: index(int): 1(before normalize), 2(after normalize)
    # augmentor(AugModule)
    def add_augmentation(self, index, augmentor):
        self.transform.transforms.insert(index, augmentor)

    # store augmented dataset
    # Args: count(int): how many times to augment, output_dir(string): directory to store augmented images
    # log_intv(int): interval to report progress
    def save_augment_dataset(self, count, augmentor, output_dir, log_intv=10):
        tr_pre = self.transform.transforms[0]  # ToTensor
        tr_after = self.transform.transforms[1]  # Normalize
        l = len(self.data)

        with torch.no_grad():
            data = []
            for idx, (img, target) in enumerate(zip(self.data, self.targets)):
                tensor_img = tr_pre(img)
                for i in range(count+1):
                    if i == 0:  # original image
                        aug_img = tensor_img
                    else:
                        aug_img = augmentor.auto_exploit(tensor_img)
                    aug_img = tr_after(aug_img)
                    data.append((aug_img, target))

                if idx % (l / log_intv) == 0 and idx != 0:
                    # save
                    batch_idx = int(idx * log_intv / l) - 1
                    torch.save(data, os.path.join(output_dir, 'batch{}'.format(batch_idx)))
                    logging.info("progress: %d / %d",  batch_idx, log_intv)
                    del data
                    data = []

            # save last batch
            batch_idx = log_intv - 1
            torch.save(data, os.path.join(output_dir, 'batch{}'.format(batch_idx)))
            logging.info("progress: %d / %d", batch_idx, log_intv)
            del data

    # make subset of dataset
    def get_subset(self, index, x, y):
        mask = np.zeros(len(y), dtype=bool)
        mask[index] = True

        return x[mask], y[mask]

    # apply class imbalance and flipped label
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

    # add augmentor module
    # Args: index(int): ignore
    # augmentor(AugModule)
    def add_augmentation(self, index, augmentor):
        self.transform.transforms.append(augmentor)


# loaded augmented dataset
class AugDataset(data.Dataset):
    def __init__(self, data_path):
        self.data, self.targets = AugDataset.load_augment_dataset(data_path)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_augment_dataset(data_path):
        x = []
        y = []

        batch_list = sorted(os.listdir(data_path))
        for idx in range(len(batch_list)):
            data = torch.load(os.path.join(data_path, batch_list[idx]))
            for (image, target) in data:
                x.append(image)
                y.append(target)

        x = torch.stack(x, dim=0)
        y = torch.tensor(y)
        return x, y
