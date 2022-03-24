import time
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from networks.nets import LeNet


# Simple classifier to evaluate dataset
class Classifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.init_models()
        self.epochs = cfg.TRAIN.epochs
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=cfg.TRAIN.lr)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.TRAIN.decay_epochs, gamma=cfg.TRAIN.decay_factor)

    # initialize classifier network
    def init_models(self):
        cfg = self.cfg
        if cfg.TRAIN.model == 'LeNet':
            model = LeNet(cfg).to(cfg.device)
        else:
            raise RuntimeError("{} Not Implemented".format(cfg.DISTILL.model))
        return model

    # train model with train_loader
    def train(self):
        model = self.model
        device = self.cfg.device
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_loader = self.cfg.train_loader

        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # evaluate trained model
    def test(self):
        model = self.model
        device = self.cfg.device
        test_loader = self.cfg.test_loader

        model.eval()
        avg_loss = 0
        accuracy = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                avg_loss += F.cross_entropy(output, label, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(label.view_as(pred)).sum().item()

        avg_loss /= len(test_loader.dataset)
        accuracy = accuracy * 100. / len(test_loader.dataset)

        return avg_loss, accuracy

    # train and evaluate model
    def train_and_evaluate(self):
        do_evaluate = self.cfg.TASK.test
        test_intv = 0
        final_loss = 0
        final_accuracy = 0
        train_time = 0  # train time per epoch

        logging.info('Start training')

        if do_evaluate:
            test_intv = self.cfg.TRAIN.test_intv
            logging.info("Evaluate model every {} epoch".format(test_intv))

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.train()
            train_time += (time.time()-t0)
            if do_evaluate and (epoch % test_intv == 0):
                loss, accu = self.test()
                logging.info('Epoch {}: Average Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, loss, accu))

            if (epoch == self.epochs) and do_evaluate:
                final_loss = loss
                final_accuracy = accu

        if do_evaluate:
            logging.info('Final Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(final_loss, final_accuracy))
        logging.info('Time cost for training: {:.2f}s per one epoch'.format(train_time/self.epochs))


# classifier using steps instead of train loader
class StepClassifier(Classifier):
    def __init__(self, cfg, steps):
        super().__init__(cfg)
        self.steps = steps

    def train(self):
        steps = self.steps
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler

        model.train()
        for step, (data, label, lr) in enumerate(steps):
            data = data.detach()
            label = label.detach()
            lr = lr.detach()

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward(lr.squeeze())
            optimizer.step()
        scheduler.step()
