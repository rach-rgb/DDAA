import time
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from augmentation import autoaug_update
from networks.nets import LeNet, AlexCifarNet


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
        elif cfg.TRAIN.model == 'AlexCifarNet':
            model = AlexCifarNet(cfg).to(cfg.device)
        else:
            logging.exception("{} Not Implemented".format(cfg.DISTILL.model))
            raise
        logging.info("Classifier Network: {}".format(cfg.TRAIN.model))
        return model

    # train model with test_train_loader
    def train(self):
        model = self.model
        device = self.cfg.device
        optimizer = self.optimizer
        scheduler = self.scheduler
        train_loader = self.cfg.test_train_loader

        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # test model with test_loader
    # use val_loader if valid=True
    def test(self, valid=False):
        model = self.model
        device = self.cfg.device
        test_loader = self.cfg.val_loader if valid else self.cfg.test_loader

        avg_loss = 0
        accuracy = 0
        model.eval()
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

    # train and test model
    def train_and_evaluate(self, valid=False, autoaug=False, modules=None):
        cfg = self.cfg
        device = cfg.device

        # model evaluation
        do_test = cfg.TASK.test or valid
        test_intv = cfg.TRAIN.test_intv if cfg.TASK.test else self.epochs + 999

        # auto-augmentation
        do_autoaug = autoaug
        search_intv = cfg.TAUG.search_intv if do_autoaug else self.epochs + 999
        if do_autoaug:
            aug_module, p_optimizer = modules
        else:
            aug_module = None
            p_optimizer = None
        assert int(do_autoaug) + int(valid) < 2

        logging.info('Start training')
        if do_test:
            logging.info("Evaluate model every {} epoch".format(test_intv))

        train_time = 0  # train time per epoch
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.train()
            train_time += (time.time() - t0)

            # explore auto-aug policy
            if do_autoaug and (epoch % search_intv == 0):
                aug_module.explore()
                search_t0 = time.time()
                autoaug_update(device, self.model, p_optimizer, cfg.val_loader)
                search_t = time.time() - search_t0
                logging.info('Epoch: {:4d}, Search time: {:.2f}'.format(epoch, search_t))
                aug_module.exploit()

            if do_test and (epoch % test_intv == 0):
                loss, accu = self.test(valid)
                logging.info('Epoch {}: Average Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, loss, accu))

        if do_test:
            final_loss, final_accu = self.test(valid)
            if valid:
                logging.info('Validation Loss: {:.4f}, Accuracy: {:.0f}%'.format(final_loss, final_accu))
            else:
                logging.info('Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(final_loss, final_accu))
        logging.info('Time cost for training: {:.2f}s per one epoch'.format(train_time / self.epochs))
