import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/09_translation_transformer",
            datetime.strftime(datetime.now(), "%Y%m%d%H%M"),
        )
        self.num_layers = 4
        self.d_model = 128
        self.num_heads = 8
        self.dff = 512
        self.dropout_rate = 0.1
        self.batch_size = 16
        self.train_epochs = 50
        # CustomSchedule parameters
        self.init_lr = 0.00001
        self.lr_after_warmup = 0.0005
        self.final_lr = 0.0001
        self.warmup_epochs = 2
        self.decay_epochs = 18