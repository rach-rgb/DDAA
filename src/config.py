import yaml
import logging


# Configuration Instance
class Config(object):
    def __init__(self, dict_config=None):
        super().__init__()
        self.set_attribute(dict_config)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as stream:
            return Config(yaml.load(stream, Loader=yaml.FullLoader))

    # Initialize dataset information
    @staticmethod
    def init_dataset(cfg):
        if cfg.name == 'MNIST':
            dataset_info = {
                'mean': (0.1307, ),
                'std': (0.3081, ),
                'num_classes': 10,
                'num_channels': 1,
                'input_size': 28,
                'labels': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            }
        elif cfg.name == 'CIFAR-10':
            dataset_info = {
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.247, 0.243, 0.261),
                'num_classes': 10,
                'num_channels': 3,
                'input_size': 32,
                'labels': ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            }
        else:
            logging.error("Dataset {} not implemented".format(cfg.name))
            raise NotImplementedError
        cfg.set_attribute(dataset_info)

    # Initialize augmentation information
    @staticmethod
    def init_augment(cfg):
        if cfg.name == 'MNIST':
            aug_info = {
                'mean': (0.1307, ),
                'std': (0.3081, ),
                'num_classes': 10,
                'num_channels': 3,
                'in_features': 84,
                'aug_list': ['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Identity']
            }
        elif cfg.name == 'CIFAR-10':
            aug_info = {
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.247, 0.243, 0.261),
                'num_classes': 10,
                'num_channels': 3,
                'in_features': 4096,
                'aug_list': ['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                             'AutoContrast', 'Invert', 'Equalize', 'Solarize', 'Posterize', 'Contrast',
                             'Color', 'Brightness', 'Sharpness', 'Flip', 'Identity']
            }
        else:
            logging.error("Augmentation for dataset {} not implemented".format(cfg.name))
            raise NotImplementedError
        cfg.set_attribute(aug_info)

    def set_attribute(self, dict_config):
        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                c = Config(dict_config[key])
                if key == 'DATA_SET':
                    Config.init_dataset(c)
                if key == 'RAUG' or key == 'DAUG' or key == 'TAUG':
                    Config.init_augment(c)
                self.__dict__[key] = c
            else:
                self.__dict__[key] = dict_config[key]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setattr__(self, key, value):
        self.set_attribute({key: value})


def search_cfg(cfg):
    data = cfg.DATA_SET.name

    # search size
    if cfg.DATA_SET.source == "raw":
        if data == 'CIFAR-10':
            cfg.DATA_SET.search_size = 45000    # (45744 / 50000)
        else:   # data == 'MNIST'
            cfg.DATA_SET.search_size = 54000    # (54000 / 60000)
    else:   # distilled
        n_steps = cfg.DISTILL.d_steps
        cfg.DATA_SET.search_size = int(n_steps * 0.9) * cfg.DISTILL.num_per_class * cfg.DATA_SET.num_classes

    # search_batch_size
    if cfg.TRAIN.model == 'AlexCifarNet':
        cfg.DATA_SET.search_batch_size = 256
    else:   # model == 'LeNet'
        cfg.DATA_SET.search_batch_size = 512
    # others: wresnet40_2 -> 32, resnet50 -> 128
