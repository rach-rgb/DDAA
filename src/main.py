import logging

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((cfg.DATA_SET.mean,), (cfg.DATA_SET.std,))
    ])
    # get train and validation dataset
    if cfg.TASK.validation is True:
        clean_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        train_idx, val_idx, _, _ = train_test_split(range(len(clean_dataset)), clean_dataset.targets,
                                                    stratify=clean_dataset.targets, test_size=cfg.DATA_SET.val_size)
        train_dataset = MessyDataset(cfg, True, clean_dataset, index=train_idx, transform=transform)
        # validation set
        val_dataset = MessyDataset(cfg, False, clean_dataset, index=val_idx, transform=transform)
        val_loader = data.DataLoader(val_dataset, cfg.DATA_SET.batch_size,
                                     num_workers=cfg.DATA_SET.num_workers, pin_memory=True, shuffle=True)
        cfg.val_loader = val_loader
    else:
        clean_dataset = datasets.MNIST(cfg.DATA_SET.root, train=True, download=True)
        train_dataset = MessyDataset(cfg, True, clean_dataset, transform=transform)

    train_loader = data.DataLoader(train_dataset, cfg.DATA_SET.batch_size,
                                   num_workers=cfg.DATA_SET.num_workers, pin_memory=True, shuffle=True)
    cfg.train_loader = train_loader

    logging.info('Load train dataset: %s, size: %d, class imbalance: %.2f, label noise: %.1f',
                 cfg.DATA_SET.name, len(cfg.train_loader.dataset), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)
    if cfg.TASK.validation is True:
        logging.info('Load validation dataset: %s, size: %d', cfg.DATA_SET.name, len(cfg.val_loader.dataset))

    # get test dataset
    test_dataset = datasets.MNIST(cfg.DATA_SET.root, train=False, transform=transform, download=True)
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
        main(Config.from_yaml('../configs/no_dd.yaml'))
    except Exception:
        logging.exception("Fatal error:")
        raise
