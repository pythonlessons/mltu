import stow
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = stow.join('Models/05_sound_to_text', datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.sample_rate = 16000
        self.frame_length = 255
        self.frame_step = 128

        self.vocab = "abcdefghijklmnopqrstuvwxyz'?! "
        self.max_text_length = None
        self.max_spectrogram_length = None


        self.batch_size = 2
        self.learning_rate = 0.001
        self.train_epochs = 1000
        self.train_workers = 20