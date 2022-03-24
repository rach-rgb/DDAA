import torch
import logging

from config import Config
from distillation import Distiller
from dataset import CustomMNISTDataset
from utils import save_results, load_results
from classification import Classifier, StepClassifier


def main(cfg):
    # get device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg.device = device

    # get train dataset
    train_dataset = CustomMNISTDataset(cfg, train=True, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.DATA_SET.batch_size)
    cfg.train_loader = train_loader
    logging.info('Load train dataset: %s, size: %d, class imbalance: %.1f, label noise: %.1f',
                 cfg.DATA_SET.name, len(train_loader.dataset), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)

    # get test dataset
    test_dataset = CustomMNISTDataset(cfg, train=False, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, cfg.DATA_SET.batch_size)
    cfg.test_loader = test_loader
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_loader.dataset))

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
            StepClassifier(cfg, steps).train_and_evaluate()
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
