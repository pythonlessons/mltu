import tensorflow as tf
from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block, activation_layer


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input", dtype=tf.float32)

    # expand dims to add channel dimension
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    # Convolution layer 1
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")

    # Convolution layer 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Dense layer
    x = layers.Dense(256)(x)
    x = activation_layer(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)

    # Classification layer
    output = layers.Dense(output_dim + 1, activation="softmax", dtype=tf.float32)(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model