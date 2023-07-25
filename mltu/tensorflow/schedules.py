import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate schedule for Transformer, but could be used for any model.

    Args:
        steps_per_epoch (int): Number of steps per epoch.
        init_lr (float, optional): Initial learning rate. Defaults to 0.00001.
        lr_after_warmup (float, optional): Learning rate after warmup. Defaults to 0.0001.
        final_lr (float, optional): Final learning rate. Defaults to 0.00001.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 15.
        decay_epochs (int, optional): Number of decay epochs. Defaults to 85.
    """
    def __init__(
            self,
            steps_per_epoch: int,
            init_lr: float=0.00001,
            lr_after_warmup: float=0.0001,
            final_lr: float=0.00001,
            warmup_epochs: int=15,
            decay_epochs: int=85,
        ):
        """ Initialize CustomSchedule."""
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = float(warmup_epochs)
        self.decay_epochs = float(decay_epochs)
        self.steps_per_epoch = steps_per_epoch

    def get_config(self) -> dict:
        """ Get config of CustomSchedule.

        Returns:
            dict: Config of CustomSchedule.
        """
        return {
            "init_lr": self.init_lr,
            "lr_after_warmup": self.lr_after_warmup,
            "final_lr": self.final_lr,
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def calculate_lr(self, epoch: int) -> float:
        """ linear warm up - linear decay 

        Args:
            epoch (int): Epoch number.

        Returns:
            float: Learning rate.
        """
        epoch = tf.cast(epoch, tf.float32)
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / self.decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step: int):
        """ Call CustomSchedule.

        Args:
            step (int): Step number.

        Returns:
            float: Learning rate.
        """
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)