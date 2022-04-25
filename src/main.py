import sys
import logging
from os import path

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import Config
from dataset import MessyDataset
from augmentation import AugModule
from distillation import Distiller
from utils import save_results, load_results
from classification import Classifier, StepClassifier


def main(cfg):
    # set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # construct dataset loader
    batch_size = cfg.DATA_SET.batch_size
    num_workers = cfg.DATA_SET.num_workers
    tr_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.DATA_SET.mean,), (cfg.DATA_SET.std,))
    ])
    if cfg.DATA_SET.train_split:
        clean_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        train_idx, val_idx, _, _ = train_test_split(range(len(clean_dataset)), clean_dataset.targets,
                                                    stratify=clean_dataset.targets, test_size=cfg.DATA_SET.val_size)
        # train set
        train_dataset = MessyDataset(cfg, clean_dataset, mess=True, index=train_idx, transform=tr_normalize)
        # validation set
        val_dataset = MessyDataset(cfg, clean_dataset, mess=False, index=val_idx, transform=tr_normalize)
        val_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        cfg.val_loader = val_loader
        logging.info('Load validation dataset: %s, size: %d', cfg.DATA_SET.name, len(val_loader.dataset))
    else:
        clean_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        train_dataset = MessyDataset(cfg, clean_dataset, mess=True, transform=tr_normalize)

    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    cfg.train_loader = train_loader
    logging.info('Load train dataset: %s, size: %d, class imbalance: %.2f, label noise: %.2f',
                 cfg.DATA_SET.name, len(train_loader.dataset), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)

    # test set
    test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=tr_normalize, download=True)
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    cfg.test_loader = test_loader
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_dataset.data))

    # apply distillation
    steps = None
    if cfg.TASK.distill is True:
        if cfg.DISTILL.load is True:
            steps = load_results(cfg)
        else:
            steps = Distiller(cfg).distill()
        if cfg.DISTILL.save_output is True:
            save_results(cfg, steps)

    # train and evaluate model
    if cfg.TASK.train:
        if cfg.TASK.distill is True:
            logging.info('Use distilled dataset with size: %d for training', len(steps))
            cls = StepClassifier(cfg)
            if cfg.TRAIN.augment:
                cls.set_step(steps, AugModule(device, cfg.T_AUGMENT))
            else:
                cls.set_step(steps)
            cls.train_and_evaluate()
        else:
            Classifier(cfg).train_and_evaluate()


if __name__ == '__main__':
    logging.basicConfig(filename='../output/logging.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        if len(sys.argv) < 2:   # default
            config_dir = 'default.yaml'
        else:
            config_dir = sys.argv[1]
        main(Config.from_yaml(path.join('../configs/', config_dir)))
    except Exception:
        logging.exception("No configuration")
        raise
