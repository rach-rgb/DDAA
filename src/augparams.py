# source: AdaAug
import torch.nn as nn


# Auto-Augment parameter generator
# stack num_layers * (num_hidden, ReLU) + FC layer
class ProjectModel(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers, num_hidden):
        super(ProjectModel, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 0:
            layers = [nn.Linear(in_feature, num_hidden), nn.ReLU()]
            for _ in range(self.num_layers-1):
                layers.append(nn.Linear(num_hidden, num_hidden))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(num_hidden, out_feature))
        else:
            layers = [nn.Linear(in_feature, out_feature)]
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection(x)
