import logging

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from config import Config
from dataset import MessyDataset
from distillation import Distiller
from utils import save_results, load_results
from classification import Classifier, StepClassifier


def main(cfg):
    # get device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # get train dataset
    transform_train = transforms.Compose([
        transforms.Normalize((cfg.DATA_SET.mean,), (cfg.DATA_SET.std,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.DATA_SET.mean,), (cfg.DATA_SET.std,))
    ])

    # train_set = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
    # val_loader = None
    # val_size = 0
    # if cfg.TASK.validation is True:     # construct validation set
    #     val_size = cfg.DATA_SET.val_size
    #     train_set, val_set = torch.utils.data.random_split(train_set, [len(train_set.data)-val_size, val_size])
    #     val_loader = data.DataLoader(MessyDataset(cfg, False, val_set, transform_train), cfg.DATA_SET.batch_size*2)
    #
    # train_dataset = MessyDataset(cfg, True, train_set, transform_train)
    # train_loader = data.DataLoader(train_dataset, cfg.DATA_SET.batch_size, num_workers=cfg.DATA_SET.num_workers,
    #                                pin_memory=True, shuffle=True)
    #
    # cfg.train_loader = train_loader
    # logging.info('Load train dataset: %s, size: %d, class imbalance: %.1f, label noise: %.1f',
    #              cfg.DATA_SET.name, len(train_set.dataset)-val_size, cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)
    # if cfg.TASK.validation is True:
    #     cfg.val_loader = val_loader
    #     logging.info('Load validation dataset: %s, size: %d',
    #                  cfg.DATA_SET.name, val_size)

    # get train dataset
    train_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, transform=transform_test, download=True)
    train_loader = data.DataLoader(train_dataset, cfg.DATA_SET.batch_size, num_workers=cfg.DATA_SET.num_workers,
                                   pin_memory=True, shuffle=True)
    cfg.train_loader = train_loader

    # get test dataset
    test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=transform_test, download=True)
    test_loader = data.DataLoader(test_dataset, cfg.DATA_SET.batch_size, num_workers=cfg.DATA_SET.num_workers,
                                  pin_memory=True, shuffle=True)
    cfg.test_loader = test_loader
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_dataset.data))

    # distillation
    steps = None
    if cfg.TASK.distill is True:
        if cfg.DISTILL.load is True:
            steps = load_results(cfg)
        else:
            logging.info('Apply dataset distillation')
            steps = Distiller(cfg).distill()

        if cfg.TASK.save_output is True:
            save_results(cfg, steps)

    # train and evaluate model
    if cfg.TASK.train is True:
        if cfg.TASK.distill is True:
            logging.info('Use distilled dataset with size: %d for training', len(steps))
            cls = StepClassifier(cfg)
            cls.set_step(steps)
            cls.train_and_evaluate()
        else:
            Classifier(cfg).train_and_evaluate()


if __name__ == '__main__':
    logging.basicConfig(filename='../output/logging.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        main(Config.from_yaml('../configs/default.yaml'))
    except Exception:
        logging.exception("Fatal error:")
        raise
