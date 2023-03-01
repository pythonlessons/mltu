class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch: int, logs=None):
        pass

    def on_train_batch_end(self, batch: int, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch: int, logs=None):
        pass

    def on_test_batch_end(self, batch: int, logs=None):
        pass

    def on_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_epoch_end(self, epoch: int, logs=None):
        pass

    def on_batch_begin(self, batch: int, logs=None):
        pass

    def on_batch_end(self, batch: int, logs=None):
        pass
