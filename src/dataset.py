import logging

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class ToFloatTensor(object):
    def __call__(self, img):
        return torch.unsqueeze(img.type(torch.FloatTensor) / 255, 0)


tr_MNIST = transforms.Compose([
    ToFloatTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

tr_CIFAR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# create messy dataset
class MessyDataset(data.Dataset):
    def __init__(self, cfg, dataset, mess, index=None, transform=None):
        x = dataset.data
        y = np.array(dataset.targets)

        if index is not None:  # subset
            x, y = self.get_subset(index, x, y)
        if mess:  # apply mess ratio for train set
            x, y = self.make_mess(cfg, x, y)

        self.data = x
        self.targets = y

        self.transform = transform
        self.is_MNIST = cfg.DATA_SET.name == 'MNIST'

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

        x = x[mask]
        y = y[mask]

        return x, y

    def make_mess(self, cfg, x, y):
        imbalance_r = cfg.DATA_SET.imbalance
        noise_r = cfg.DATA_SET.noise
        num_classes = cfg.DATA_SET.num_classes
        assert 0 <= imbalance_r <= 1
        assert 0 <= noise_r <= 1

        # apply class imbalance
        if imbalance_r != 1:
            logging.info("Apply class imbalance ratio: %0.2f", imbalance_r)
            small_label = [x for x in range(0, int(num_classes / 2))]
            large_size = int(len(x) / num_classes)
            small_size = int(large_size * imbalance_r)

            remove_idx = []
            ny = y.numpy()
            for label in small_label:
                remove_idx = remove_idx + list(np.where(ny == label)[0][:large_size - small_size])

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
    def __init__(self, steps, cfg=None):
        if cfg is not None:
            nc = cfg.DATA_SET.num_channels
            mean = cfg.DATA_SET.mean, std = cfg.DATA_SET.std
        else:
            nc = 1
            mean = 0.1307
            std = 0.3081

        self.data, self.targets = self.convert(mean, std, nc, steps)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def convert(self, mean, std, nc, steps):
        # convert torch to nparray
        if isinstance(steps[0][0], torch.Tensor):
            np_steps = []
            for data, label, lr in steps:
                np_data = data.detach().permute(0, 2, 3, 1).to('cpu').numpy()
                np_label = label.detach().to('cpu').numpy()
                if lr is not None:
                    lr = lr.detach().cpu().numpy()
                np_steps.append((np_data, np_label, lr))
            steps = np_steps

        # generate PIL image
        x = []
        y = []
        for i, (data, labels, lr) in enumerate(steps):
            for n, (img, label) in enumerate(zip(data, labels)):
                if nc == 1:
                    img = img[..., 0]
                img = ((img * std + mean).clip(0, 1) * 255).astype(np.uint8)
                x.append(img)
                y.append(label)
        return x, y

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
