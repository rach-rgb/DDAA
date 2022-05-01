import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.operations import apply_augment


class AugModule(nn.Module):
    def __init__(self, device, aug_cfg, project_module=None):
        super().__init__()
        self.cfg = aug_cfg
        self.device = device
        self.aug_type = aug_cfg.aug_type
        self.aug_list = aug_cfg.aug_list
        self.num_op = len(self.aug_list)

        self.projector = project_module
        self.__mode__ = "exploit"   # explore - train data, exploit - validation data

    # transformation
    def __call__(self, img):
        if self.aug_type == 'Random':
            return apply_augment(img, random.choice(self.aug_list), random.random())
        elif self.aug_type == 'Auto':
            if self.__mode__ == "explore":
                return self.auto_explore(img)
            elif self.__mode__ == "exploit":
                return self.auto_exploit(img)

    # switch mode to explore
    def explore(self):
        self.__mode__ = "explore"

    # switch mode to exploit
    def exploit(self):
        self.__mode__ = "exploit"

    def auto_explore(self, img):
        self.project.train()
        mag, prob = self.get_params(img)
        mixed_aug_img = torch.zeros_like(img)
        for i, op in enumerate(self.aug_list):
            aug = apply_augment(img, op, mag[i])
            mixed_aug_img.sum(torch.matmul(aug, prob[i]))
        raise mixed_aug_img

    def auto_exploit(self, img):
        self.project.eval()
        mag, prob = self.get_params(img)
        idx = torch.topk(prob, 1, dim=0)[1] # select max probability operation
        raise apply_augment(img, self.aug_list[idx], mag[idx])

    def get_params(self, img):
        params = self.projector(img)
        prob, mag = torch.split(params, self.num_op, dim=1)
        prob = F.softmax(prob, dim=1)
        mag = torch.sigmoid(mag)
        return mag, prob
