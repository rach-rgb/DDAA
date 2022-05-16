import os
import random
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from augparams import Projector
from aug_operations import apply_augment
from networks.nets import LeNet, AlexCifarNet


# Return Auto-Augmentation Module and Auto-Aug Parameter Optimizer
# Args: device(str): CPU or CUDA, aug_cfg(Config): configuration for augmentation
# ext(Object): extractor, p(Object): projector
# Returns: Augmentation Module, Parameter Optimizer
def autoaug_creator(device, aug_cfg, ext, p=None):
    if p is None:
        p = Projector(aug_cfg.in_features, 2 * len(aug_cfg.aug_list)).to(device)
        loaded = False
    else:
        loaded = True

    augmentor = AugModule(device, aug_cfg, loaded, ext, p)
    p_optimizer = torch.optim.Adam(
        p.parameters(),
        lr=aug_cfg.p_lr,
        betas=(0.9, 0.999),
        weight_decay=aug_cfg.p_decay_factor
    )
    return augmentor, p_optimizer


# Update Projection Module
# Args: device(str): CPU or CUDA, augmentor, p_optimizer
# val_loader(Dataloader): dataloader of validation set
def autoaug_update(device, augmentor, p_optimizer, val_loader):
    vdata, vlabel = next(iter(val_loader))
    if not vdata.is_cuda:
        vdata = vdata.to(device)
    vlabel = vlabel.to(device)

    augmentor.explore()
    p_optimizer.zero_grad()
    output = augmentor.auto_explore(vdata)
    loss = F.cross_entropy(output, vlabel)
    loss.backward()
    p_optimizer.step()
    del output, loss

    augmentor.exploit()


# Save Trained Feature Extractor and Parameter Projector
# Args: aug_cfg(Config): configuration for augmentation, augmentor
def autoaug_save(aug_cfg, augmentor):
    output_dir = os.path.join(Path(os.getcwd()).parent, aug_cfg.output_dir)

    # save task_model (feature extractor)
    output1 = os.path.join(output_dir, 'extractor_weights.pth')
    torch.save(augmentor.extractor.state_dict(), output1)

    # save projector (parameter generator)
    output2 = os.path.join(output_dir, 'projector_weights.pth')
    torch.save(augmentor.projector.state_dict(), output2)

    logging.info('AutoAug Task Model and Projector saved to {}'.format(aug_cfg.output_dir))


# Load Feature Extractor and Projector and Create Auto-Augmentation Module
# Args: device(str): CPU or CUDA, cfg(Config): configuration for entire project,
# aug_cfg(Config): configuration for augmentation
# Returns: Auto-Augmentation Module and Project Optimizer
def autoaug_load(device, cfg, aug_cfg):
    load_dir = os.path.join(Path(os.getcwd()).parent, aug_cfg.load_dir)

    # load task_model
    path1 = os.path.join(load_dir, 'extractor_weights.pth')
    if aug_cfg.name == 'MNIST':
        f = LeNet(cfg).to(device)
    else:   # CIFAR-10
        f = AlexCifarNet(cfg).to(device)
    f.load_state_dict(torch.load(path1))

    # load projector
    path2 = os.path.join(load_dir, 'projector_weights.pth')
    p = Projector(aug_cfg.in_features, 2 * len(aug_cfg.aug_list)).to(device)
    p.load_state_dict(torch.load(path2))

    logging.info('AutoAug Module loaded from {}'.format(aug_cfg.load_dir))

    return autoaug_creator(device, aug_cfg, ext=f, p=p)


def perturb_param(param, delta):
    if delta <= 0:
        return param

    amt = random.uniform(0, delta)
    if random.random() < 0.5:
        return max(0, param - amt)
    else:
        return min(1, param + amt)


# Augmentation Module
class AugModule(nn.Module):
    def __init__(self, device, aug_cfg, loaded=False, ext=None, p=None):
        super().__init__()
        self.cfg = aug_cfg
        self.device = device
        self.aug_type = aug_cfg.aug_type
        self.aug_list = aug_cfg.aug_list
        self.n_ops = len(self.aug_list)
        self.random_apply = aug_cfg.random_apply
        self.k_ops = aug_cfg.k_ops
        self.temp = aug_cfg.temp
        self.delta = aug_cfg.delta

        # auto aug
        self.__mode__ = "exploit"   # explore - train data, exploit - validation data
        self.loaded = loaded  # module is loaded i.e, already trained
        self.extractor = ext
        self.projector = p

    # transformation
    def __call__(self, img):
        if self.random_apply:   # return raw image
            if random.random() > 0.5:
                return img

        if self.aug_type == 'Random':
            return apply_augment(img, random.choice(self.aug_list), random.random())
        elif self.aug_type == 'Auto':
            if self.__mode__ == "explore":
                return img  # perform exploration manually
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
        with torch.no_grad():
            prob, mag = self.get_params(img.to(self.device))
            op_indices = torch.topk(prob, self.k_ops, dim=0)[1]
            if self.k_ops > 1:
                aimg = img
                for idx in op_indices:
                    mag = perturb_param(mag[idx].item(), self.delta)
                    aimg = apply_augment(aimg, self.aug_list[idx], mag[idx].item())
                return aimg
            else:
                idx = op_indices
                return apply_augment(img, self.aug_list[idx], mag[idx].item())

    # return classification result of mixed feature validation image
    # Args: imgs(Tensor): validation batch
    # Returns: output(Tensor): classification result
    def auto_explore(self, imgs):
        vfeat = self.get_mixed_feat(imgs)
        output = self.extractor.cls_label(vfeat)
        del vfeat
        return output

    # return mixed augment features
    def get_mixed_feat(self, imgs):
        self.projector.train()
        mixed_feats = []
        for img in imgs:
            prob, mag = self.get_params(img)
            aug_imgs = []
            for i, op in enumerate(self.aug_list):
                aug_imgs.append(apply_augment(img, op, mag[i].item()).to(self.device))
            aug_feats = self.extractor.get_feature(torch.stack(aug_imgs, dim=0))
            mixed_feats.append(torch.matmul(prob, aug_feats))
        return torch.stack(mixed_feats, dim=0)

    # predict augmentation parameter
    def get_params(self, img):
        self.extractor.eval()
        if self.__mode__ == "exploit":
            T = self.temp
        else:   # mode == 'explore'
            T = 1.0

        feature = self.extractor.get_feature(img.unsqueeze(0))  # make batch
        params = self.projector(feature)
        prob, mag = torch.split(params, self.n_ops, dim=1)
        prob = F.softmax(prob, dim=1).squeeze(0)
        mag = torch.sigmoid(mag/T).squeeze(0)
        return prob, mag
