import yaml


class Config(object):
    def __init__(self, dict_config=None):
        super().__init__()
        self.set_attribute(dict_config)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as stream:
            return Config(yaml.load(stream, Loader=yaml.FullLoader))

    # set information of dataset
    @staticmethod
    def init_dataset(cfg):
        if cfg.name == 'MNIST':
            dataset_info = {
                'mean': 0.1307,
                'std': 0.3081,  # normalization
                'num_classes': 10,
                'num_channels': 1,
                'input_size': 28,
                'labels': ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            }
        else:
            raise RuntimeError("Dataset {} not implemented".format(cfg.name))
        cfg.set_attribute(dataset_info)

    def set_attribute(self, dict_config):
        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                c = Config(dict_config[key])
                if key == 'DATA_SET':
                    Config.init_dataset(c)
                self.__dict__[key] = c
            else:
                self.__dict__[key] = dict_config[key]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setattr__(self, key, value):
        self.set_attribute({key: value})