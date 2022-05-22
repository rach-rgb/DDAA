# CB_loss source: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
import logging

import torch
import numpy as np
import torch.nn.functional as F


# Args: output(tensor): model output, label(tensor): target, ty(str): type of loss
# reduction(str): F.cross_entropy reduction method, params(tuple): parameter for class balanced focal loss
# Returns: calculated loss
def get_loss(output, label, ty, reduction='mean', params=None):
    if ty == 'CE':  # cross entropy
        return F.cross_entropy(output, label, reduction=reduction)
    elif ty == 'BF':  # class balanced focal loss
        device, n_classes, n_per_classes = params
        return CB_loss(label, output, n_per_classes, n_classes, 'FL', device)
    elif 'BCE' in ty:  # class blanced cross entropy
        device, n_classes, n_per_classes = params
        return CB_loss(label, output, n_per_classes, n_classes, ty, device)
    else:
        logging.error("Loss Model {} not implemented".format(ty))
        raise NotImplementedError


def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    fl = torch.sum(weighted_loss)

    fl /= torch.sum(labels)
    return fl


# class balanced loss
def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, device, beta=0.9999, gamma=1.0):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0).to(device)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "FL":
        return focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "BCE-sig":
        return F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "BCE-soft":
        pred = logits.softmax(dim=1)
        return F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

