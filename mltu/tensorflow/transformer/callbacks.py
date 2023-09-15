import tensorflow as tf
from keras.callbacks import Callback
from mltu.tensorflow.callbacks import Model2onnx


class EncDecSplitCallback(Callback):
    """Callback to extract the encoder and decoder models from Transformer model and save them separately
    Also, this callback incorporates Model2onnx callback to convert the encoder and decoder models to ONNX format

    Args:
        model_path (str): Path to save the encoder and decoder models
        encoder_metadata (dict, optional): Metadata to save with the encoder model. Defaults to None.
        decoder_metadata (dict, optional): Metadata to save with the decoder model. Defaults to None.
    """

    def __init__(
        self,
        model_path: str,
        encoder_metadata: dict = None,
        decoder_metadata: dict = None,
        model_name = "model.h5"
    ):
        """Callback to extract the encoder and decoder models from Transformer model and save them separately"""
        super(EncDecSplitCallback, self).__init__()
        self.model_path = model_path
        self.encoder_metadata = encoder_metadata
        self.decoder_metadata = decoder_metadata
        self.model_name = model_name

    def on_train_end(self, epoch: int, logs: dict = None):
        try:
            # load best model weights
            self.model.load_weights(self.model_path + "/" + self.model_name)
            
            # extract encoder and decoder models
            encoder_model = tf.keras.Model(
                inputs=self.model.inputs[0], outputs=self.model.get_layer("encoder").output
            )
            decoder_model = tf.keras.Model(
                inputs=[self.model.inputs[1], self.model.get_layer("encoder").output],
                outputs=self.model.layers[-1].output,
            )

            # save encoder and decoder models
            encoder_model.save(self.model_path + "/encoder.h5")
            decoder_model.save(self.model_path + "/decoder.h5")

            # convert encoder and decoder models to onnx
            Model2onnx.model2onnx(encoder_model, self.model_path + "/encoder.onnx")
            Model2onnx.model2onnx(decoder_model, self.model_path + "/decoder.onnx")

            # save encoder and decoder metadata
            if self.encoder_metadata:
                Model2onnx.include_metadata(self.model_path + "/encoder.onnx", self.encoder_metadata)
            if self.decoder_metadata:
                Model2onnx.include_metadata(self.model_path + "/decoder.onnx", self.decoder_metadata)
        except Exception as e:
            print(e)
            pass
