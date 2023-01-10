import stow
import yaml

class BaseModelConfigs:
    def __init__(self):
        self.model_path = None

    def serialize(self):
        # get object attributes
        return self.__dict__

    def save(self, name: str='configs.yaml'):
        if self.model_path is None:
            raise Exception("Model path is not specified")

        stow.mkdir(self.model_path)
        with open(stow.join(self.model_path, name), 'w') as f:
            yaml.dump(self.serialize(), f)

    @staticmethod
    def load(configs_path: str):
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        config = BaseModelConfigs()
        for key, value in configs.items():
            setattr(config, key, value)

        return config