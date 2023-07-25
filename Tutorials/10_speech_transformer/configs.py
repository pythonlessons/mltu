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

        self.batch_size = 32
        # self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20

        self.num_layers_encoder = 6
        self.num_layers_decoder = 2
        self.d_model = 128
        self.num_heads = 8
        self.dff = 128
        self.dropout_rate = 0.1
