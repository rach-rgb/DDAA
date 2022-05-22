import os
import sys
import logging
from pathlib import Path

import torch
from torchvision import datasets

from config import Config
import custom_dataset.transform as tr
from custom_dataset.dataset import RawDataset
from custom_augment.augmentation import AugModule, autoaug_load


def main(cfg, count):
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # construct dataset loader
    is_MNIST = cfg.DATA_SET.name == 'MNIST'

    if is_MNIST:
        total_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        tr_train = tr.train_MNIST
    else:  # CIFAR-10
        total_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=True, download=True)
        tr_train = tr.train_CIFAR

    train_dataset = RawDataset(cfg, total_dataset, mess=False, transform=tr_train)
    logging.info('Load train dataset: %s, size: %d', cfg.DATA_SET.name, len(train_dataset))

    if cfg.TAUG.aug_type == "Random":
        logging.info("Apply Random Augmentation")
        augmentor = AugModule(device, cfg.TAUG)
    else:   # cfg.TAUG.aug_type == "Auto"
        logging.info("Apply Auto Augmentation")
        # model load
        assert cfg.TAUG.load  # use pretrained augment policy
        augmentor, p_optimizer = autoaug_load(device, cfg, cfg.TAUG)

    output_dir = os.path.join(Path(os.getcwd()).parent, 'output', 'augment-' + cfg.DATA_SET.name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset.save_augment_dataset(count, augmentor, output_dir)
    logging.info("%s augmented %d times stored at %s", cfg.DATA_SET.name, count, str(output_dir))


if __name__ == '__main__':
    assert len(sys.argv) >= 2
    cfg_file = sys.argv[1]
    log_file = cfg_file.split('.')[0] + '-augment-log.txt'
    if len(sys.argv) >= 3:
        count = sys.argv[2]
    else:
        count = 3

    logging.basicConfig(filename=os.path.join('../output/', log_file), level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    try:
        torch.multiprocessing.set_start_method('spawn')
        main(Config.from_yaml(os.path.join('../configs/', cfg_file)), count)
    except Exception:
        logging.exception("Terminate by error")
