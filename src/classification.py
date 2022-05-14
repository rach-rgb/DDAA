import time
import logging

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from loss_model import get_loss
from augmentation import autoaug_update
from networks.nets import LeNet, AlexCifarNet


# Simple classifier to evaluate dataset
class Classifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.epochs = cfg.TRAIN.epochs
        self.model = self.init_models()
        self.optimizer, self.scheduler, self.loss_info = self.init_others(cfg)

    # initialize classifier network
    def init_models(self):
        cfg = self.cfg
        if cfg.TRAIN.model == 'LeNet':
            model = LeNet(cfg).to(self.device)
        elif cfg.TRAIN.model == 'AlexCifarNet':
            model = AlexCifarNet(cfg).to(self.device)
        else:
            logging.error("Model {} not implemented".format(cfg.DISTILL.model))
            raise NotImplementedError
        logging.info("Classifier Network: {}".format(cfg.TRAIN.model))
        return model

    def init_others(self, cfg):
        # optimizer
        if cfg.TRAIN.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(self.model.parameters(), lr=cfg.TRAIN.lr)
        elif cfg.TRAIN.optimizer == 'SGD':
            c = cfg.TRAIN.SGD
            optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.TRAIN.lr,
                                        weight_decay=c.weight_decay, momentum=c.momentum)
        else:
            logging.error("Optimizer {} not implemented".format(cfg.TRAIN.optimizer))
            raise NotImplementedError

        # scheduler
        if cfg.TRAIN.scheduler == 'StepLR':
            c = cfg.TRAIN.StepLR
            scheduler = StepLR(self.optimizer, step_size=c.decay_epochs, gamma=c.decay_factor)
        elif cfg.TRAIN.scheduler == 'ReduceLR':
            c = cfg.TRAIN.ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=c.factor)
        else:
            logging.error("Scheduler {} not implemented".format(cfg.TRAIN.scheduler))
            raise NotImplementedError

        # loss
        if cfg.TRAIN.tloss_crit == 'CE':
            loss_info = None
        elif cfg.TRAIN.tloss_crit == 'BF':
            loss_info = (
            cfg.device, cfg.test_train_loader.dataset.n_classes, cfg.test_train_loader.dataset.n_per_classes)
        else:
            logging.error("Loss Model {} not implemented".format(cfg.TRAIN.tloss_crit))
            raise NotImplementedError

        logging.info("Optimizer: {}, Scheduler: {}, Loss: {}"
                     .format(cfg.TRAIN.optimizer, cfg.TRAIN.scheduler, cfg.TRAIN.tloss_crit))

        return optimizer, scheduler, loss_info

    # train model with test_train_loader
    def train(self):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        scheduler = self.scheduler
        loss_type = self.cfg.TRAIN.tloss_crit
        train_loader = self.cfg.test_train_loader

        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = get_loss(output, label, loss_type, params=self.loss_info)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # test model with test_loader
    def test(self):
        model = self.model
        device = self.device
        test_loader = self.cfg.test_loader

        avg_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                avg_loss += get_loss(output, label, "CE", reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(label.view_as(pred)).sum().item()

        avg_loss /= len(test_loader.dataset)
        accuracy = accuracy * 100. / len(test_loader.dataset)

        return avg_loss, accuracy

    # train and test classifier
    # Args: autoaug(Bool): apply auto-augmentation or not, modules(tuple): models used for auto-aug
    def train_and_evaluate(self, autoaug=False, modules=None):
        cfg = self.cfg
        device = self.device
        do_autoaug = autoaug
        do_test = cfg.TASK.test

        unreach_ep = self.epochs + 1
        ex_intv = cfg.TAUG.search_intv if do_autoaug else unreach_ep
        test_intv = cfg.TRAIN.test_intv if cfg.TASK.test else unreach_ep

        # auto-augmentation
        if do_autoaug:
            aug_module, p_optimizer = modules
        else:
            aug_module = None
            p_optimizer = None

        logging.info('Start training')
        if do_test:
            logging.info("Evaluate model every {} epoch".format(test_intv))

        loss = 0
        accu = 0

        train_t0 = time.time()  # train time per epoch
        for epoch in range(1, self.epochs + 1):
            self.train()

            # explore auto-aug policy
            if do_autoaug and (epoch % ex_intv == 0):
                if not aug_module.loaded:
                    ex_t0 = time.time()
                    autoaug_update(device, aug_module, p_optimizer, cfg.val_loader)
                    ex_t = time.time() - ex_t0
                    logging.info('Epoch: {:4d}, Search time: {:.2f}'.format(epoch, ex_t))

            if do_test and (epoch % test_intv == 0):
                loss, accu = self.test()
                logging.info('Epoch {}: Average Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, loss, accu))
            # end of for loop

        train_t = time.time() - train_t0
        if do_test:
            logging.info('Test Loss: {:.4f}, Accuracy: {:.0f}%'.format(loss, accu))

        logging.info('Time cost for training: {:.2f}s per one epoch'.format(train_t / self.epochs))
