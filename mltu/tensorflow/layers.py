import tensorflow as tf
from keras import layers

class SelfAttention(layers.Layer):
    """  A self-attention layer for convolutional neural networks.
    
    This layer takes as input a tensor of shape (batch_size, height, width, channels)
    and applies self-attention to the channels dimension.

    Args:
        num_heads (int): The number of attention heads to use. Defaults to 8.
        wrapper (tf.keras.layers.Wrapper): A wrapper layer to apply to the convolutional layers.

    Raises:
        TypeError: If `wrapper` is provided and is not a subclass of `tf.keras.layers.Wrapper`.
    """
    def __init__(self, num_heads: int = 8, wrapper: tf.keras.layers.Wrapper = None):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.wrapper = wrapper

        if wrapper and not issubclass(wrapper, tf.keras.layers.Wrapper):
            raise TypeError("wrapper must be a class derived from tf.keras.layers.Wrapper")

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
        })
        return config

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.query_conv = self._conv(filters=c // self.num_heads)
        self.key_conv = self._conv(filters=c // self.num_heads)
        self.value_conv = self._conv(filters=c)
        self.gamma = self.add_weight("gamma", shape=[1], initializer=tf.zeros_initializer(), trainable=True)

    def _conv(self, filters: int) -> tf.keras.layers.Layer:
        """ Helper function to create a convolutional layer with the given number of filters.

        Args:
            filters (int): The number of filters to use.

        Returns:
            tf.keras.layers.Layer: The created convolutional layer.
        """
        conv = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')
        if self.wrapper:
            conv = self.wrapper(conv)

        return conv

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Apply the self-attention mechanism to the input tensor.

        Args:
            inputs (tf.Tensor): The input tensor of shape (batch_size, height, width, channels).

        Returns:
            tf.Tensor: The output tensor after the self-attention mechanism is applied.
        """
        _, h, w, c = inputs.shape
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)

        q_reshaped = tf.reshape(q, [-1, h * w, c // self.num_heads])
        k_reshaped = tf.reshape(k, [-1, h * w, c // self.num_heads])
        v_reshaped = tf.reshape(v, [-1, h * w, c])

        # Compute the attention scores by taking the dot product of the query and key tensors.
        attention_scores = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)

        # Scale the attention scores by the square root of the number of channels.
        attention_scores = attention_scores / tf.sqrt(tf.cast(c // self.num_heads, dtype=tf.float32))

        # Apply a softmax function to the attention scores to obtain the attention weights.
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        # Apply the attention weights to the value tensor to obtain the attention output.
        attention_output = tf.matmul(attention_weights, v_reshaped)

        # Reshape the attended value tensor to the original input tensor shape.
        attention_output = tf.reshape(attention_output, [-1, h, w, c])

        # Apply the gamma parameter to the attended value tensor and add it to the output tensor.
        attention_output = self.gamma * attention_output + inputs

        return attention_output