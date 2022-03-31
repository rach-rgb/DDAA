import logging

import torch
import numpy as np
import torch.utils.data as data


# create messy dataset
class MessyDataset(data.Dataset):
    def __init__(self, cfg, train, subset, transform=None):
        x = subset.dataset.data
        y = subset.dataset.targets

        if train is True:   # apply mess ratio for train set
            x, y = self.make_mess(cfg, x, y)
        num_channels = cfg.DATA_SET.num_channels
        input_size = cfg.DATA_SET.input_size

        self.data = x.view(len(x), num_channels, input_size, input_size).float()
        self.targets = y

        self.transform = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.data[idx])
        else:
            img = self.data[idx]
        return img, self.targets[idx]

    def __len__(self):
        return len(self.data)

    def make_mess(self, cfg, x, y):
        imbalance_r = cfg.DATA_SET.imbalance
        noise_r = cfg.DATA_SET.noise
        num_classes = cfg.DATA_SET.num_classes
        assert 0 <= imbalance_r <= 1
        assert 0 <= noise_r <= 1

        # apply class imbalance
        if imbalance_r != 1:
            logging.info("Apply class imbalance ratio: %f", imbalance_r)
            small_label = [x for x in range(0, int(num_classes/2))]
            small_size = int(len(x) / len(num_classes) * imbalance_r)

            remove_idx = []
            ny = y.numpy()
            for label in small_label:
                remove_idx = remove_idx + list(np.where(ny == label)[0][:small_size])

            mask = np.ones(len(y), dtype=bool)
            mask[remove_idx] = False

            x = x[mask]
            y = y[mask]

        # apply label noise
        if noise_r != 0:
            logging.info("Apply noise ratio: %f", noise_r)
            noise = torch.randint(num_classes-1, size=(int(len(y) * noise_r),))
            y = torch.cat([y[:len(y) - int(len(y) * noise_r)], noise])

        return x, y
