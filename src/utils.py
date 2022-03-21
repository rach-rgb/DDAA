import yaml


class Config(object):
    def __init__(self, dict_config=None):
        super().__init__()
        self.set_attribute(dict_config)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as stream:
            return Config(yaml.load(stream, Loader=yaml.FullLoader))

    def set_attribute(self, dict_config):
        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                self.__dict__[key] = Config(dict_config[key])
            else:
                self.__dict__[key] = dict_config[key]

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setattr__(self, key, value):
        self.set_attribute({key:value})
