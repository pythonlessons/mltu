import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/10_wav2vec2_torch",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        self.batch_size = 8
        self.train_epochs = 60
        self.train_workers = 20

        self.init_lr = 1.0e-8
        self.lr_after_warmup = 1e-05
        self.final_lr = 5e-06
        self.warmup_epochs = 10
        self.decay_epochs = 40
        self.weight_decay = 0.005
        self.mixed_precision = True

        self.max_audio_length = 246000 
        self.max_label_length = 256

        self.vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']