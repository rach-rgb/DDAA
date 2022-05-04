# source: https://github.com/jamestszhim/adaptive_augment
import torch.nn as nn
import torch.nn.functional as F


# Auto-Augment parameter generator
# stack num_layers * (num_hidden, ReLU) + FC layer
class Projector(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=0, num_hidden=128):
        super(Projector, self).__init__()

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
        return self.projection(x)
