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
        self.do_auto = cfg.AUGMENT.auto_aug    # perform auto-augmentation
        self.aug_list = cfg.AUGMENT.aug_list
        self.num_aug = len(self.aug_list)
        self.p_model = p_model

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

            aug_images.append(apply_augment(pil_img, self.aug_list[oid], mag).ToTensor())
        return torch.stack(aug_images, dim=0).cuda(self.device)

    def auto_aug(self, images):
        # TODO: implement this
        # set p_model.train() and p_model.eval() properly
        # invoke projection model to get instance-wise magnitude and probability
        raise NotImplementedError
