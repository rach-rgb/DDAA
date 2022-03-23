import torch, logging
from utils import Config
from distillation import Trainer
from networks.nets import LeNet
from dataset import CustomMNISTDataset
from post_process import save_results


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
                 cfg.DATA_SET.name, len(train_loader), cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)

    # get test dataset
    test_dataset = CustomMNISTDataset(cfg, train=False, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, cfg.DATA_SET.batch_size)
    cfg.test_loader = test_loader
    logging.info('Load test dataset: %s, size: %d', cfg.DATA_SET.name, len(test_loader))

    # distillation
    if cfg.TASK.distill is True:
        logging.info('Apply dataset distillation')
        steps = Trainer(cfg).train()

        if cfg.TASK.save_output is True:
            save_results(cfg, steps)

    # train model
    if cfg.TASK.train is True:
        logging.info('Train model')
        if cfg.TAST.distill is True:
            logging.info('Use distilled dataset for training')

    # test model
    if cfg.TASK.test is True:
        if cfg.TAST.train is False:
            raise RuntimeError("Cannot test model w/o training")
        logging.info('Test model')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        main(Config.from_yaml('../configs/default.yaml'))
    except Exception:
        logging.exception("Fatal error:")
        raise
