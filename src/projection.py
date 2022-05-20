# source: https://github.com/jamestszhim/adaptive_augment
import torch.nn as nn


# Auto Augmentation parameter generator
class Projector(nn.Module):
    def __init__(self, in_features, out_features, n_layers=0, n_hidden=128):
        super(Projector, self).__init__()
        self.n_layers = n_layers
        if self.n_layers > 0:
            layers = [nn.Linear(in_features, n_hidden), nn.ReLU()]
            for _ in range(self.n_layers-1):
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden, out_features))
        else:
            layers = [nn.Linear(in_features, out_features)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)
