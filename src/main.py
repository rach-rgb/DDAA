import torch, logging
from utils import Config
from networks.networks import LeNet
from dataset import CustomMNISTDataset
from distillation import Trainer


def main(cfg):
    logging.info('Dataset Distillation')

    device = torch.device("cuda")
    cfg.device = device

    dataset = CustomMNISTDataset('../data', train=True, download=True,
                                 imbalance_r=cfg.DATA_SET.imbalance, noise_r=cfg.DATA_SET.noise)
    train_loader = torch.utils.data.DataLoader(dataset, cfg.DATA_SET.batch_size)
    cfg.train_loader = train_loader
    logging.info('Load dataset: %s, class imbalance: %.1f, label noise: %.1f',
                 cfg.DATA_SET.name, cfg.DATA_SET.imbalance, cfg.DATA_SET.noise)

    if cfg.TASK.model == 'LeNet':
        task_model = LeNet(cfg).to(device)
    else:
        raise RuntimeError("{} Not Implemented".format(cfg.TASK.model))

    Trainer(task_model, cfg).train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try:
        main(Config.from_yaml('../configs/base.yaml'))
    except Exception:
        logging.exception("Fatal error:")
        raise
