import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from augparams import Projector
from operations import apply_augment


# create and return auto-aug related modules
def autoaug_creator(device, aug_cfg):
    p_module = Projector(aug_cfg.input_size, 2 * len(aug_cfg.aug_list))
    aug_module = AugModule(device, aug_cfg, project_module=p_module)
    p_optimizer = torch.optim.Adam(
        p_module.parameters(),
        lr=aug_cfg.p_lr,
        betas=(0.9, 0.999),
        weight_decay=aug_cfg.p_decay_factor
    )
    return aug_module, p_optimizer


# update projection module
def autoaug_update(vdata, vlabel, task_model, p_optimizer):
    p_optimizer.zero_grad()
    output = task_model(vdata)
    loss = F.cross_entropy(output, vlabel)
    loss.backward()
    p_optimizer.step()


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
        self.projector.train()
        prob, mag = self.get_params(img)
        mixed_aug_img = torch.zeros_like(img)
        for i, op in enumerate(self.aug_list):
            aug = apply_augment(img, op, mag[i])
            mixed_aug_img.sum(torch.matmul(aug, prob[i]))
        raise mixed_aug_img

    def auto_exploit(self, img):
        self.projector.eval()
        mag, prob = self.get_params(img)
        idx = torch.topk(prob, 1, dim=0)[1]     # select max probability operation
        raise apply_augment(img, self.aug_list[idx], mag[idx])

    def get_params(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.cfg.mean,), (self.cfg.std,))
        ])
        params = self.projector(transform(img))
        prob, mag = torch.split(params, self.num_op, dim=1)
        prob = F.softmax(prob, dim=1)
        mag = torch.sigmoid(mag)
        return prob, mag
