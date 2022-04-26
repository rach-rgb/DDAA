import random

import torch
import torch.nn as nn
from torchvision import transforms

from src.operations import apply_augment


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

    def augment_steps(self, steps):
        aug_steps = []
        for data, label, lr in steps:
            data = data.detach()
            label = label.detach()
            lr = lr.detach()

            aug_steps.append((data, label, lr))
            for i in range(0, self.num_data):
                aug_steps.append((self.augment(data), label, lr))
        return aug_steps

    def __call__(self, img):
        if self.aug_type == 'Random':
            return apply_augment(img, random.choice(self.aug_list), random.random())
        elif self.aug_type == 'Auto':
            return self.auto_aug(img)

    def augment(self, images):
        if self.aug_type == 'Random':
            return self.rand_aug_tensor(images)
        elif self.aug_type == 'Auto':
            return self.auto_aug(images)

    # apply random augmentation with random magnitude
    def rand_aug_tensor(self, images):
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
