import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/10_speech_transformer",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        self.frame_length = 256
        self.frame_step = 160
        self.fft_length = 384

        self.input_shape = (2048, 193)
        self.max_spectrogram_length = 2048

        self.batch_size = 128
        self.train_epochs = 1000
        self.train_workers = 20

        self.num_layers_encoder = 2
        self.num_layers_decoder = 2
        self.d_model = 512
        self.num_heads = 8
        self.dff = 2048
        self.dropout_rate = 0.1


        self.init_lr = 0.000001
        self.lr_after_warmup = 0.0003
        self.final_lr = 0.0001
        self.warmup_epochs = 3
        self.decay_epochs = 40