# CB_loss source: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
import logging

import torch
import numpy as np
import torch.nn.functional as F


# Args: output(tensor): model output, label(tensor): target, ty(str): type of loss
# reduction(str): F.cross_entropy reduction method, params(tuple): parameter for class balanced focal loss
# Returns: calculated loss
def get_loss(output, label, ty, reduction='none', params=None):
    if ty == 'CE':  # cross entropy
        return F.cross_entropy(output, label, reduction=reduction)
    elif ty == 'BF':    # class balanced focal loss
        n_classes, n_per_classes = params
        return CB_loss(output, label, n_classes, n_per_classes)
    else:
        logging.error("Loss Model {} not implemented".format(ty))
        raise NotImplementedError


def CB_loss(logits, labels, n_classes, n_per_classes, beta=0.9999, gamma=2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      n_per_classes: A python list of size [no_of_classes].
      n_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, n_per_classes)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * n_classes

    labels_one_hot = F.one_hot(labels, n_classes).float()

    weights = (torch.tensor(weights).float()).unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1).unsqueeze(1)
    alpha = weights.repeat(1, n_classes)

    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels_one_hot * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))
    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels_one_hot)
    return focal_loss
