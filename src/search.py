import sys
import logging
from os import path

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import custom_dataset.transform as tr
from utils import load_results
from config import Config, search_cfg
from classification import Classifier
from custom_dataset.dataset import RawDataset, StepDataset
from custom_augment.augmentation import autoaug_creator, autoaug_save


def main(cfg):
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    search_cfg(cfg)

    # dataloader
    n_workers = cfg.DATA_SET.num_workers
    is_MNIST = cfg.DATA_SET.name == 'MNIST'
    is_raw = cfg.DATA_SET.source == 'raw'

    if is_raw:
        if is_MNIST:
            total_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
            tr_train = tr.train_MNIST
            tr_search_pre = tr.valid_pre_MNIST
        else:  # CIFAR-10
            total_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=True, download=True)
            tr_train = tr.train_CIFAR
            tr_search_pre = tr.valid_pre_CIFAR
        train_idx, search_idx, _, _ = train_test_split(range(len(total_dataset)), total_dataset.targets,
                                                       stratify=total_dataset.targets,
                                                       test_size=cfg.DATA_SET.search_size)
        train_dataset = RawDataset(cfg, total_dataset, mess=False, index=train_idx, transform=tr_train)
        search_dataset = RawDataset(cfg, total_dataset, mess=False, index=search_idx, transform=tr_search_pre)
    else:  # distilled dataset
        steps = load_results(cfg)[:cfg.DISTILL.d_steps]
        train_n_step = int(cfg.DATA_SET.search_size / cfg.DISTILL.num_per_class / cfg.DATA_SET.num_classes)
        train_dataset = StepDataset(cfg.DATA_SET.num_classes, steps[:train_n_step])
        search_dataset = StepDataset(cfg.DATA_SET.num_classes, steps[train_n_step:])

    cfg.test_train_loader = DataLoader(train_dataset, cfg.DATA_SET.batch_size, shuffle=True, drop_last=False,
                                       num_workers=n_workers)
    cfg.val_loader = DataLoader(search_dataset, cfg.DATA_SET.search_batch_size, shuffle=True, drop_last=False,
                                num_workers=n_workers)
    logging.info('Load train dataset: %s, size: %d', cfg.DATA_SET.name, len(train_dataset))
    logging.info('Load search dataset: %s, size: %d', cfg.DATA_SET.name, len(search_dataset))

    if is_MNIST:
        test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=tr.test_MNIST, download=True)
    else:  # CIFAR-10
        test_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=False, transform=tr.test_CIFAR, download=True)
    cfg.test_loader = DataLoader(test_dataset, cfg.DATA_SET.batch_size, shuffle=True, num_workers=n_workers,
                                 pin_memory=True)
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_dataset))

    # model
    cls = Classifier(cfg)
    augmentor, p_optimizer = autoaug_creator(device, cfg.TAUG, cls.model)
    cfg.test_train_loader.dataset.add_augmentation(1, augmentor)
    cls.train_and_evaluate(autoaug=True, modules=(augmentor, p_optimizer))

    # model save
    if cfg.TAUG.save:
        autoaug_save(cfg.TAUG, augmentor)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    cfg_file = sys.argv[1]
    log_file = cfg_file.split('.')[0] + '-search-log.txt'

    logging.basicConfig(filename=path.join('../output/', log_file), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    try:
        # torch.multiprocessing.set_start_method('spawn')
        main(Config.from_yaml(path.join('../configs/search/', cfg_file)))
    except Exception:
        logging.exception("Terminate by error")
