# source: https://github.com/jamestszhim/adaptive_augment
import torch.nn as nn
import torch.nn.functional as F


# Auto-Augment parameter generator
# stack num_layers * (num_hidden, ReLU) + FC layer
class Projector(nn.Module):
    def __init__(self, in_feature, out_feature, num_channels=1, num_layers=0, num_hidden=128):
        super(Projector, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 6, 5, padding=2 if in_feature == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.num_layers = num_layers
        if self.num_layers > 0:
            layers = [nn.Linear(84, num_hidden), nn.ReLU()]
            for _ in range(self.num_layers-1):
                layers.append(nn.Linear(num_hidden, num_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(num_hidden, out_feature))
        else:
            layers = [nn.Linear(in_feature, out_feature)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(1, -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        return self.projection(out)
