import os
import tf2onnx
import onnx
from keras.callbacks import Callback

import logging

class Model2onnx(Callback):
    """ Converts the model to onnx format after training is finished. """
    def __init__(
        self, 
        saved_model_path: str, 
        metadata: dict=None
        ) -> None:
        """ Converts the model to onnx format after training is finished.
        Args:
            saved_model_path (str): Path to the saved .h5 model.
            metadata (dict, optional): Dictionary containing metadata to be added to the onnx model. Defaults to None.
        """
        super().__init__()
        self.saved_model_path = saved_model_path
        self.metadata = metadata

    def on_train_end(self, logs=None):
        self.model.load_weights(self.saved_model_path)
        self.onnx_model_path = self.saved_model_path.replace(".h5", ".onnx")
        tf2onnx.convert.from_keras(self.model, output_path=self.onnx_model_path)

        if self.metadata and isinstance(self.metadata, dict):
            # Load the ONNX model
            onnx_model = onnx.load(self.onnx_model_path)

            # Add the metadata dictionary to the model's metadata_props attribute
            for key, value in self.metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = value

            # Save the modified ONNX model
            onnx.save(onnx_model, self.onnx_model_path)

class TrainLogger(Callback):
    """Logs training metrics to a file.
    
    Args:
        log_path (str): Path to the directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'logs.log'.
        logLevel (int, optional): Logging level. Defaults to logging.INFO.
    """
    def __init__(self, log_path: str, log_file: str='logs.log', logLevel=logging.INFO) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.log_file))
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict=None):
        epoch_message = f"Epoch {epoch}; "
        logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)