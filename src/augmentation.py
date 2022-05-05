import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from augparams import Projector
from operations import apply_augment


# create and return auto-aug related modules
def autoaug_creator(device, aug_cfg, cls):
    p_module = Projector(aug_cfg.in_features, 2 * len(aug_cfg.aug_list)).to(device)
    aug_module = AugModule(device, aug_cfg, project_module=p_module, task_model=cls)
    p_optimizer = torch.optim.Adam(
        p_module.parameters(),
        lr=aug_cfg.p_lr,
        betas=(0.9, 0.999),
        weight_decay=aug_cfg.p_decay_factor
    )
    return aug_module, p_optimizer


# update projection module
def autoaug_update(device, task_model, p_optimizer, val_loader):
    vfeat, vlabel = next(iter(val_loader))
    vlabel = vlabel.to(device)

    p_optimizer.zero_grad()
    output = task_model.get_label(vfeat)
    loss = F.cross_entropy(output, vlabel)
    loss.backward()
    p_optimizer.step()


class AugModule(nn.Module):
    def __init__(self, device, aug_cfg, task_model=None, project_module=None):
        super().__init__()
        self.cfg = aug_cfg
        self.device = device
        self.aug_type = aug_cfg.aug_type
        self.aug_list = aug_cfg.aug_list
        self.n_ops = len(self.aug_list)

        # auto aug
        self.__mode__ = "exploit"   # explore - train data, exploit - validation data
        self.model = task_model
        self.projector = project_module

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

    # return augment image
    def auto_exploit(self, img):
        self.projector.eval()
        prob, mag = self.get_params(img)
        idx = torch.topk(prob, 1, dim=0)[1]     # select max probability operation
        return apply_augment(img, self.aug_list[idx], mag[idx])

    # return mixed augment features
    def auto_explore(self, img):
        tr_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.cfg.mean,), (self.cfg.std,))
        ])

        self.projector.train()
        prob, mag = self.get_params(img)
        mixed_feat = torch.zeros(self.cfg.in_features)
        for i, op in enumerate(self.aug_list):
            aug_img = apply_augment(img, op, mag[i])
            aug_feat = self.model.get_feature(tr_norm(aug_img)[None, :].to(self.device))
            torch.add(mixed_feat, torch.mul(aug_feat, prob[i]))
        return mixed_feat

    # predict augmentation parameter
    def get_params(self, img):
        tr_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.cfg.mean,), (self.cfg.std,))
        ])

        self.model.eval()
        feature = self.model.get_feature(tr_norm(img)[None, :].to(self.device))
        params = self.projector(feature)
        prob, mag = torch.split(params, self.n_ops, dim=1)
        prob = F.softmax(prob, dim=1).squeeze()
        mag = torch.sigmoid(mag).squeeze()
        return prob, mag
