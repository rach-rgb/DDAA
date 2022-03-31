import torch.nn as nn
import torch.nn.functional as F
from .reparam_module import ReparamModule


class LeNet(ReparamModule):
    def __init__(self, cfg):
        if cfg.DATA_SET.name != 'MNIST':
            raise RuntimeError("Cannot use dataset {} for LeNet".format(cfg.DATA_SET.name))
        num_classes = cfg.DATA_SET.num_classes
        num_channels = cfg.DATA_SET.num_channels
        input_size = cfg.DATA_SET.input_size

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5, padding=2 if input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if num_classes <= 2 else num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out



