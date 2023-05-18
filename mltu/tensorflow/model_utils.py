import typing
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

class CustomModel(Model):
    """ Custom TensorFlow model for debugging training process purposes
    """
    def train_step(self, train_data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs, targets = train_data
        with tf.GradientTape() as tape:
            results = self(inputs, training=True)
            loss = self.compiled_loss(targets, results, regularization_losses=self.losses)
            gradients = tape.gradient(loss, self.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(targets, results)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, test_data):
        inputs, targets = test_data
        # Get prediction from model
        results = self(inputs, training=False)

        # Update the loss
        self.compiled_loss(targets, results, regularization_losses=self.losses)

        # Update the metrics
        self.compiled_metrics.update_state(targets, results)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def activation_layer(layer, activation: str="relu", alpha: float=0.1) -> tf.Tensor:
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions
    Args:
        layer: tf.Tensor
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)
    Returns:
        tf.Tensor
    """
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer


def residual_block(
        x: tf.Tensor,
        filter_num: int,
        strides: typing.Union[int, list] = 2,
        kernel_size: typing.Union[int, list] = 3,
        skip_conv: bool = True,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        activation: str = "relu",
        dropout: float = 0.2):
    # Create skip connection tensor
    x_skip = x

    # Perform 1-st convolution
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    # Perform 2-nd convoluti
    x = layers.Conv2D(filter_num, kernel_size, padding = padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)

    # Perform 3-rd convolution if skip_conv is True, matchin the number of filters and the shape of the skip connection tensor
    if skip_conv:
        x_skip = layers.Conv2D(filter_num, 1, padding = padding, strides = strides, kernel_initializer=kernel_initializer)(x_skip)

    # Add x and skip connection and apply activation function
    x = layers.Add()([x, x_skip])     
    x = activation_layer(x, activation=activation)

    # Apply dropout
    if dropout:
        x = layers.Dropout(dropout)(x)

    return x