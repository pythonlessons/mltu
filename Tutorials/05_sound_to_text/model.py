import tensorflow as tf
from keras import layers
from keras.models import Model

from mltu.model_utils import residual_block, activation_layer

def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # expand dims to add channel dimension
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    rnn_layers = 5
    rnn_units = 128
    # Expand the dimension to use 2D CNN.
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(input)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    # x = layers.ReLU(name="conv_1_relu")(x)
    x = activation_layer(x, activation='leaky_relu')
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    # x = layers.ReLU(name="conv_2_relu")(x)
    x = activation_layer(x, activation='leaky_relu')
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            # activation="tanh",
            # recurrent_activation="sigmoid",
            # use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    # x = layers.ReLU(name="dense_1_relu")(x)
    x = activation_layer(x, activation='leaky_relu')
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)




    # Convolution layer 1
    # x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    # x = layers.BatchNormalization()(x)
    # x = activation_layer(x, activation='leaky_relu')

    # # Convolution layer 2
    # x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    # x = layers.BatchNormalization()(x)
    # x = activation_layer(x, activation='leaky_relu')
    
    # # Reshape the resulted volume to feed the RNNs layers
    # x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # # RNN layers
    # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, reset_after=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, reset_after=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, reset_after=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, reset_after=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.GRU(128, return_sequences=True, reset_after=True))(x)
    # x = layers.Dropout(dropout)(x)

    # # Dense layer
    # x = layers.Dense(units=256)(x)
    # x = activation_layer(x, activation='leaky_relu')
    # x = layers.Dropout(dropout)(x)

    # Classification layer
    # output = layers.Dense(units=output_dim + 1, activation="softmax")(x)

    # x1 = residual_block(input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    # x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    # x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    # x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    # x7 = residual_block(x6, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)

    # x8 = residual_block(x7, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    # x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    # squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    # blstm = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(squeezed)
    # blstm = layers.Dropout(dropout)(blstm)

    # blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(blstm)
    # blstm = layers.Dropout(dropout)(blstm)

    # output = layers.Dense(output_dim + 1, activation='softmax', name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model