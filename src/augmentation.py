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
        self.aug_fix = aug_cfg.aug_fix  # 0: fix (*), 1: fix (epoch, label, *)
        self.aug_list = aug_cfg.aug_list
        self.num_op = len(self.aug_list)
        self.num_data = aug_cfg.aug_num
        self.p_layer = p_layer

        # operation history for aug_fix = 1
        self.op_cache = {}

        # log selected operation
        self.log = aug_cfg.log
        if self.log:
            self.op_count = {}
            for i in range(0, self.num_op):
                self.op_count[i] = 0

    def augment(self, images, info=None):
        if self.aug_fix == 1:
            if self.aug_type == 'Random':
                return self.rand_aug_fixed(images, info)
            elif self.aug_type == 'Auto':
                logging.exception("Not Implemented")
                raise

        if self.aug_type == 'Random':
            return self.rand_aug(images)
        elif self.aug_type == 'Auto':
            return self.auto_aug(images)

    # select one augmentation operation randomly
    # operation magnitude selected random
    def rand_aug(self, images):
        aug_images = []
        for image in images:
            pil_img = transforms.ToPILImage()(image)

            # select one augmentation operation
            oid = int(random.random() * self.num_op)
            if oid == self.num_op:
                oid = oid - 1
            mag = random.random()
            if self.log:
                self.op_count[oid] = self.op_count[oid] + 1

            aug_img = transforms.ToTensor()(apply_augment(pil_img, self.aug_list[oid], mag))
            aug_images.append(self.stop_gradient(aug_img.to(self.device), mag))
        return torch.stack(aug_images, dim=0)

    def rand_aug_fixed(self, images, info):
        labels = info

        aug_images = []
        for image, label in zip(images, labels):
            pil_img = transforms.ToPILImage()(image)

            # select one augmentation operation
            if labels in self.op_hist:
                oid, mag = self.op_hist[labels]
            else:
                oid = int(random.random() * self.num_op)
                if oid == self.num_op:
                    oid = oid - 1
                mag = random.random()
                self.op_hist[labels] = (oid, mag)
                if self.log:
                    self.op_count[oid] = self.op_count[oid] + 1

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

    def reset_op(self):
        self.op_cache = {}

    def log_history(self):
        for k, v in self.op_count.items():
            logging.info("%s: %d", self.aug_list[k], v)
