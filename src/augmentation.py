import logging
import random

import torch
import torch.nn as nn
from torchvision import transforms

from operations import apply_augment


class AugModule(nn.Module):
    def __init__(self, device, aug_cfg, p_layer=None):
        super().__init__()
        self.cfg = aug_cfg
        self.device = device
        self.aug_type = aug_cfg.aug_type
        self.aug_list = aug_cfg.aug_list
        self.num_op = len(self.aug_list)
        self.num_data = aug_cfg.aug_num
        self.p_layer = p_layer

        # log selected operation
        self.log = aug_cfg.log
        self.count = {}
        for i in range(0, self.num_op):
            self.count[i] = 0

    def augment(self, images):
        if self.aug_type == 'Random':
            return self.rand_aug(images)
        elif self.aug_type == 'Auto':
            return self.auto_aug(images)

    # select one augmentation operation randomly
    # operation magnitude selected random
    def rand_aug(self, images):
        aug_images = []
        for idx, image in enumerate(images):
            pil_img = transforms.ToPILImage()(image)

            # select one augmentation operation
            oid = int(random.random() * self.num_op)
            if oid == self.num_op:
                oid = oid - 1
            mag = random.random()
            if self.log:
                self.count[oid] = self.count[oid] + 1

            aug_img = transforms.ToTensor()(apply_augment(pil_img, self.aug_list[oid], mag))
            aug_images.append(self.stop_gradient(aug_img.to(self.device), mag))
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
        for k, v in self.count.items():
            logging.info("%s: %d", self.aug_list[k], v)
