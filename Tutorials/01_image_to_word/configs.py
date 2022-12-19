import stow
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = stow.join('Models/1_image_to_word', datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.height = 32
        self.width = 128
        self.max_text_length = 23
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = 20