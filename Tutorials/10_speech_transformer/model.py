import tensorflow as tf

from mltu.tensorflow.transformer.layers import EncoderLayer, Decoder, PositionalEmbedding, positional_encoding

class SpeechFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model=64):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(d_model, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.conv2 = tf.keras.layers.Conv1D(d_model, kernel_size=3, strides=2, padding="same", use_bias=False)
        self.conv3 = tf.keras.layers.Conv1D(d_model, kernel_size=3, strides=2, padding="same", use_bias=False)
        # self.max_pooling = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.linear = tf.keras.layers.Dense(d_model)

        # expand dims to add channel dimension
        # self.input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))# (inputs)

        # Convolution layer 1
        self.conv2d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)# (input)
        self.lr1 = tf.keras.layers.LeakyReLU() 
        # x = layers.BatchNormalization()(x)
        # x = activation_layer(x, activation="leaky_relu")

        # Convolution layer 2
        self.conv2d2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[2, 2], padding="same", use_bias=False)# (x)
        self.lr2 = tf.keras.layers.LeakyReLU()
        self.lr3 = tf.keras.layers.LeakyReLU()
        self.gelu1 = tf.keras.layers.Activation("gelu")
        self.gelu2 = tf.keras.layers.Activation("gelu")
        self.gelu3 = tf.keras.layers.Activation("gelu")
        # x = layers.BatchNormalization()(x)
        # x = activation_layer(x, activation="leaky_relu")
        
        # Reshape the resulted volume to feed the RNNs layers
        # self.reshape = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))# (x)


    def call(self, x):
        # x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
        # x = self.conv2d1(x)
        # x = self.bn(x)
        # x = self.lr1(x)
        # x = self.conv2d2(x)
        # x = self.bn2(x)
        # x = self.lr2(x)
        # x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # x = self.linear(x)

        x = self.conv1(x)
        x = self.bn(x)
        # x = self.lr1(x)
        x = self.gelu1(x)
        # x = self.max_pooling(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.lr2(x)
        x = self.gelu2(x)
        # x = self.max_pooling(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.lr3(x)
        x = self.gelu3(x)
        # x = self.linear(x)
        x = self.dropout(x)
        # x = self.max_pooling(x)
        return x
    
    
class Encoder(tf.keras.layers.Layer):
    """
    A custom TensorFlow layer that implements the Encoder. This layer is mostly used in the Transformer models 
    for natural language processing tasks, such as machine translation, text summarization or text classification.

    Methods:
        call: Performs the forward pass of the layer.

    Attributes:
        d_model (int): The dimensionality of the model.
        num_layers (int): The number of layers in the encoder.
        pos_embedding (PositionalEmbedding): The positional embedding layer.
        enc_layers (list): The list of encoder layers.
        dropout (tf.keras.layers.Dropout): The dropout layer.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, dropout_rate: float=0.1, activation: str="relu"):
        """
        Constructor of the Encoder.

        Args:
            num_layers (int): The number of layers in the encoder.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of heads in the multi-head attention layer.
            dff (int): The dimensionality of the feed-forward layer.
            vocab_size (int): The size of the vocabulary.
            dropout_rate (float): The dropout rate.
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.speech_embedding = SpeechFeatureEmbedding(d_model=d_model)
        self.pos_embedding = PositionalEmbedding(vocab_size=None, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate,
                        activation=activation)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        The call function that performs the forward pass of the layer.
        
        Args:
            x (tf.Tensor): The input sequence of shape (batch_size, seq_length).

        Returns:
            tf.Tensor: The output sequence of shape (batch_size, seq_length, d_model).
        """
        x = self.pos_embedding(x)  
        # here x has shape `(batch_size, seq_len, d_model)`

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.

def SpeechTransformer(
    target_vocab_size: int, 
    encoder_input_shape: int = None,
    decoder_input_shape: int = None,
    num_layers_encoder: int=6,
    num_layers_decoder: int=6,
    d_model: int=512, 
    num_heads: int=8,
    dff: int=2048,
    dropout_rate: float=0.1,
    activation: str="relu"
    ) -> tf.keras.Model:
    """
    A custom TensorFlow model that implements the Transformer architecture.

    Args:
        target_vocab_size (int): The size of the target vocabulary.
        encoder_input_size (int): The size of the encoder input sequence.
        decoder_input_size (int): The size of the decoder input sequence.
        num_layers (int): The number of layers in the encoder and decoder.
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of heads in the multi-head attention layer.
        dff (int): The dimensionality of the feed-forward layer.
        dropout_rate (float): The dropout rate.

    Returns:
        A TensorFlow Keras model.
    """
    inputs = [
        tf.keras.layers.Input(shape=encoder_input_shape, dtype=tf.float32), 
        tf.keras.layers.Input(shape=decoder_input_shape, dtype=tf.int64)
        ]
    
    encoder_input, decoder_input = inputs

    speech_embedding_layer = SpeechFeatureEmbedding(d_model=d_model)(encoder_input)
    encoder = Encoder(num_layers=num_layers_encoder, d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate, activation=activation)(speech_embedding_layer)
    decoder = Decoder(num_layers=num_layers_decoder, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate, activation=activation)(decoder_input, encoder)

    output = tf.keras.layers.Dense(target_vocab_size, dtype=tf.float32)(decoder)

    return tf.keras.Model(inputs=inputs, outputs=output)


# import numpy as np

# # vocab_size = 1000
# d_model = 512

# # # embedding_layer = PositionalEmbedding(vocab_size, d_model)

# # # random_input = np.random.randint(0, vocab_size, size=(1, 100))



# speech_embedding = SpeechFeatureEmbedding(d_model=d_model)
# pos_embedding = PositionalEmbedding(vocab_size=0, d_model=d_model, embedding=speech_embedding)

# input_shape = (1392, 193)

# random_input = np.random.randn(1, 1392, 193)

# output = pos_embedding(random_input)