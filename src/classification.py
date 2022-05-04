import time
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from networks.nets import LeNet
from augmentation import autoaug_update


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
        train_loader = self.cfg.test_train_loader

        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            return
        scheduler.step()

    # evaluate trained model
    def test(self, valid=False):
        model = self.model
        device = self.cfg.device
        if valid:
            test_loader = self.cfg.val_loader
        else:
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
    def train_and_evaluate(self, valid=False, autoaug=False, p_optimizer=None):
        cfg = self.cfg

        # model evaluation
        do_test = cfg.TASK.test
        test_intv = cfg.TRAIN.test_intv if do_test else self.epochs + 999

        # auto-augmentation
        do_autoaug = autoaug
        search_intv = cfg.TAUG.search_intv if do_autoaug else self.epochs + 999
        if do_autoaug:
            valid = False
            if p_optimizer is None:
                logging.exception("No Projection Module")
                raise

        # results
        final_loss = 0
        final_accuracy = 0
        loss = 0
        accu = 0

        logging.info('Start training')
        if do_test:
            logging.info("Evaluate model every {} epoch".format(test_intv))

        train_time = 0  # train time per epoch
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.train()
            train_time += (time.time() - t0)

            if do_autoaug and (epoch % search_intv == 0):
                search_t0 = time.time()
                vdata, vlabel = next(iter(cfg.val_loader))
                autoaug_update(vdata, vlabel, self.model, p_optimizer)
                search_t = time.time() - search_t0
                logging.info('Epoch: {:4d}, Search time: {:.2f}'.format(epoch, search_t))

            if do_test and (epoch % test_intv == 0):
                loss, accu = self.test(valid)
                logging.info('Epoch {}: Average Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, loss, accu))

            if (epoch == self.epochs) and do_test:
                final_loss = loss
                final_accuracy = accu

        if do_test:
            if valid:
                logging.info('Validation Loss: {:.4f}, Accuracy: {:.0f}%'.format(final_loss, final_accuracy))
            else:
                logging.info('Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(final_loss, final_accuracy))
        logging.info('Time cost for training: {:.2f}s per one epoch'.format(train_time / self.epochs))



# classifier using steps instead of train loader
class StepClassifier(Classifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.steps = None

    def set_step(self, steps):
        self.steps = steps
