import sys
import logging
from os import path

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import Config
from distillation import Distiller
from classification import Classifier
from utils import load_results, save_results
from dataset import tr_MNIST, tr_CIFAR, RawDataset, StepDataset
from augmentation import AugModule, autoaug_creator, autoaug_load, autoaug_save


def main(cfg):
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # construct dataset loader
    batch_size = cfg.DATA_SET.batch_size
    num_workers = cfg.DATA_SET.num_workers
    is_MNIST = cfg.DATA_SET.name == 'MNIST'

    tr_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATA_SET.mean, cfg.DATA_SET.std)
    ])

    if is_MNIST:
        clean_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        tr = tr_MNIST
    else:   # CIFAR-10
        clean_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=True, download=True)
        tr = tr_CIFAR

    if cfg.DATA_SET.train_split:
        train_idx, val_idx, _, _ = train_test_split(range(len(clean_dataset)), clean_dataset.targets,
                                                    stratify=clean_dataset.targets, test_size=cfg.DATA_SET.val_size)
        # validation loader
        val_dataset = RawDataset(cfg, clean_dataset, mess=False, index=val_idx, transform=tr)
        cfg.val_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers)
        logging.info('Load validation dataset: %s, size: %d', cfg.DATA_SET.name, len(val_dataset))
        # train loader
        train_dataset = RawDataset(cfg, clean_dataset, mess=True, index=train_idx, transform=tr)
    else:
        train_dataset = RawDataset(cfg, clean_dataset, mess=True, transform=tr)

    cfg.train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    logging.info('Load train dataset: %s, size: %d, class imbalance: %.2f, label noise: %.2f',
                 cfg.DATA_SET.name, len(train_dataset), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)

    # test loader
    if is_MNIST:
        test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=tr_norm, download=True)
    else:   # CIFAR-10
        test_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=False, transform=tr_norm, download=True)
    cfg.test_loader = data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_dataset))

    # distillation
    steps = None
    if cfg.TASK.distill:
        if cfg.DISTILL.load:
            steps = load_results(cfg)
        else:
            steps = Distiller(cfg).distill()
            if cfg.DISTILL.save_output is True:
                save_results(cfg, steps)

    # train and evaluate
    if cfg.TASK.train:
        # dataloader
        if cfg.TASK.distill:
            if not cfg.TRAIN.use_full_steps:
                steps = steps[:cfg.DISTILL.d_steps]
            step_dataset = StepDataset(cfg.DATA_SET.num_classes, steps)
            cfg.test_train_loader = data.DataLoader(step_dataset, 1, shuffle=True, num_workers=1)
            logging.info('Use distilled dataset with size: %d for training', len(steps))
        else:
            cfg.test_train_loader = cfg.train_loader  # use train loader (raw data)
        cls = Classifier(cfg)
        # augmentation
        if cfg.TRAIN.augment:
            if cfg.TAUG.aug_type == "Random":
                logging.info("Apply Random Augmentation")
                augmentor = AugModule(device, cfg.TAUG)
                cfg.test_train_loader.dataset.transform.transforms.append(augmentor)
                cls.train_and_evaluate()
            elif cfg.TAUG.aug_type == "Auto":
                logging.info("Apply Auto Augmentation")
                # model load
                if cfg.TAUG.load:   # use pretrained augment policy
                    augmentor, p_optimizer = autoaug_load(device, cfg, cfg.TAUG)
                else:
                    augmentor, p_optimizer = autoaug_creator(device, cfg.TAUG, ext=cls.model)
                cfg.test_train_loader.dataset.transform.transforms.append(augmentor)
                cls.train_and_evaluate(autoaug=True, modules=(augmentor, p_optimizer))
                # model save
                if cfg.TAUG.save:
                    autoaug_save(cfg.TAUG, augmentor)
            else:
                logging.error("Augmentation {} not implemented".format(cfg.TAUG.aug_type))
                raise NotImplementedError
        else:
            cls.train_and_evaluate()


if __name__ == '__main__':
    logging.basicConfig(filename='../output/logging.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        if len(sys.argv) < 2:  # default
            config_dir = 'default.yaml'
        else:
            config_dir = sys.argv[1]
        # torch.multiprocessing.set_start_method('spawn')
        main(Config.from_yaml(path.join('../configs/', config_dir)))
    except Exception:
        logging.exception("Terminate by error")
