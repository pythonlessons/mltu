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

        # self.vocab = "abcdefghijklmnopqrstuvwxyz'?! "
        self.input_shape = None
        # self.max_text_length = None
        self.max_spectrogram_length = None

        self.batch_size = 64
        # self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20

        self.num_layers_encoder = 1 # 6
        self.num_layers_decoder = 1 # 2
        self.d_model = 512
        self.num_heads = 2 # 8
        self.dff = 2048 # 128
        self.dropout_rate = 0.1

        # CustomSchedule parameters
        self.init_lr = 0.00001
        self.lr_after_warmup = 0.0005
        self.final_lr = 0.0001
        self.warmup_epochs = 2
        self.decay_epochs = 18
