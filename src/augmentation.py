import logging
import random

import torch
import torch.nn as nn
from torchvision import transforms

from operations import apply_augment


class AutoAug(nn.Module):
    def __init__(self, cfg, p_model=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.do_auto = cfg.AUGMENT.do_auto    # perform auto-augmentation
        self.do_manual = cfg.AUGMENT.do_manual
        self.aug_list = cfg.AUGMENT.aug_list
        if self.aug_list:
            self.aug_list = cfg.AUGMENT.man_aug_list
        self.num_aug = len(self.aug_list)
        self.p_model = p_model

        # log augmentation strategy
        self.log = cfg.AUGMENT.log
        self.hist = {}
        for i in range(0, self.num_aug):
            self.hist[i] = 0

    def augment(self, images):
        if self.do_auto:
            return self.auto_aug(images)
        else:
            return self.rand_aug(images)

    # select one augmentation operation randomly
    # operation magnitude selected random
    def rand_aug(self, images):
        aug_images = []
        for image in images:
            pil_img = transforms.ToPILImage()(image)

            # select one augmentation operation
            oid = int(random.random() * self.num_aug)
            if oid == self.num_aug:
                oid = oid - 1
            mag = random.random()
            if self.log:
                self.hist[oid] = self.hist[oid] + 1

            aug_img = transforms.ToTensor()(apply_augment(pil_img, self.aug_list[oid], mag))
            aug_images.append(self.stop_gradient(aug_img.to(self.cfg.device), mag))
        return torch.stack(aug_images, dim=0)

    def auto_aug(self, images):
        # TODO: implement this
        # set p_model.train() and p_model.eval() properly
        # invoke projection model to get instance-wise magnitude and probability
        raise NotImplementedError

    def stop_gradient(self, aug_img, magnitude):
        images = aug_img
        adds = 0

        images = images - magnitude
        adds = adds + magnitude
        images = images.detach() + adds
        return images

    def log_history(self):
        for k, v in self.hist.items():
            logging.info("%s: %d", self.aug_list[k], v)
