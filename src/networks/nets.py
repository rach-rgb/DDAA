import torch.nn as nn
import torch.nn.functional as F

from .reparam_module import ReparamModule


class LeNet(ReparamModule):
    def __init__(self, cfg):
        input_size = cfg.DATA_SET.input_size
        n_classes = cfg.DATA_SET.num_classes
        n_channels = cfg.DATA_SET.num_channels
        assert input_size == 28

        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def __f__(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        return out

    def __g__(self, x):
        out = self.fc3(x)
        return out


class AlexCifarNet(ReparamModule):
    def __init__(self, cfg):
        input_size = cfg.DATA_SET.input_size
        num_classes = cfg.DATA_SET .num_classes
        num_channels = cfg.DATA_SET.num_channels
        assert input_size == 32 and num_channels == 3

        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def __f__(self, x):
        x = self.features(x)
        return x.view(x.size(0), 4096)

    def __g__(self, x):
        return self.classifier(x)
