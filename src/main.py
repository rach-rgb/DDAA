import os, sys, logging

import torch
import torch.utils.data as data
from torchvision import datasets

from config import Config
import custom_dataset.transform as tr
from distillation import Distiller
from classification import Classifier
from utils import load_results, save_results
from custom_dataset.dataset import StepDataset, get_dataset
from custom_augment.augmentation import AugModule, autoaug_creator, autoaug_load, autoaug_save


def main(cfg):
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # construct dataset loader
    batch_size = cfg.DATA_SET.batch_size
    num_workers = cfg.DATA_SET.num_workers

    # train loader, test loader
    train_dataset, val_dataset = get_dataset(cfg)
    cfg.train_loader = data.DataLoader(train_dataset, batch_size,
                                       shuffle=True, num_workers=num_workers, pin_memory=True)
    logging.info('Load train dataset: %s, size: %d, class imbalance: %.2f, label noise: %.2f',
                 cfg.DATA_SET.name, len(train_dataset), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)
    if val_dataset is not None:
        cfg.val_loader = data.DataLoader(val_dataset, batch_size, shuffle=True, num_workers=num_workers)
        logging.info('Load validation dataset: %s, size: %d', cfg.DATA_SET.name, len(val_dataset))

    # test loader
    if cfg.DATA_SET.name == 'MNIST':
        test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=tr.test_MNIST, download=True)
    else:   # CIFAR-10
        test_dataset = datasets.CIFAR10(cfg.DATA_SET.root, train=False, transform=tr.test_CIFAR, download=True)
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
            logging.info('Use distilled dataset with size: %d for training',
                         len(steps * cfg.DATA_SET.num_classes * cfg.DISTILL.num_per_class))
        else:
            cfg.test_train_loader = cfg.train_loader  # use train loader (raw data)
        cls = Classifier(cfg)
        # augmentation
        if cfg.TRAIN.augment:
            if cfg.TAUG.aug_type == "Random":
                logging.info("Apply Random Augmentation")
                augmentor = AugModule(device, cfg.TAUG)
                cfg.test_train_loader.dataset.add_augmentation(1, augmentor)
                cls.train_and_evaluate()
            elif cfg.TAUG.aug_type == "Auto":
                logging.info("Apply Auto Augmentation")
                # model load
                if cfg.TAUG.load:   # use pretrained augment policy
                    augmentor, p_optimizer = autoaug_load(device, cfg, cfg.TAUG)
                else:
                    augmentor, p_optimizer = autoaug_creator(device, cfg.TAUG, ext=cls.model)
                cfg.test_train_loader.dataset.add_augmentation(1, augmentor)
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
        torch.multiprocessing.set_start_method('spawn')
        main(Config.from_yaml(os.path.join('../configs/', config_dir)))
    except Exception:
        logging.exception("Terminate by error")
