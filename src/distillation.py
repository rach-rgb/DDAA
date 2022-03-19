import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import logging


class Trainer(object):
    def __init__(self, model, cfg):
        self.model = model
        self.device = cfg.device
        self.epochs = cfg.TASK.epochs
        self.train_loader = cfg.train_loader
        self.optimizer = optim.Adadelta(model.parameters(), lr=cfg.TASK.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=cfg.TASK.decay_factor)

    def forward(self, model, data, label):
        model.train()
        _data, _label = data.to(self.device), label.to(self.device)

        output = model(_data)
        return F.nll_loss(output, _label)

    def backward(self, loss):
        loss.backward()

    def prefetch_train_loader_iter(self):
        device = self.device
        train_iter = iter(self.train_loader)
        for epoch in range(self.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < self.epochs - 1:
                    train_iter = iter(self.train_loader)
                yield epoch, it, val

    def train(self):
        for epoch, it, (data, label) in self.prefetch_train_loader_iter():
            if it == 0:
                self.scheduler.step()

            self.optimizer.zero_grad()
            loss = self.forward(self.model, data, label)
            self.backward(loss)

            self.optimizer.step()

            if it == 0:
                logging.info('Train Epoch: {}'.format(epoch))
