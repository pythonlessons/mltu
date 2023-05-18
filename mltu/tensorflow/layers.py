import tensorflow as tf
from keras import layers
from keras import backend as K

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
        conv = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")
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
    
    
class SpectralNormalization(tf.keras.layers.Wrapper):
    """Spectral Normalization Wrapper. !!! This is not working yet !!!"""
    def __init__(self, layer, power_iterations=1, eps=1e-12, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)

        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self.eps = eps
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                "Please initialize `TimeDistributed` layer with a "
                "`Layer` instance. You passed: {input}".format(input=layer))

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        # self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
        #                          initializer=tf.initializers.TruncatedNormal(stddev=0.02),
        #                          trainable=False,
        #                          name="sn_v",
        #                          dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name="sn_u",
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def l2normalize(self, v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
    
    def power_iteration(self, W, u, rounds=1):
        _u = u

        for _ in range(rounds):
            # v_ = tf.matmul(_u, tf.transpose(W))
            # v_hat = self.l2normalize(v_)
            _v = self.l2normalize(K.dot(_u, K.transpose(W)), eps=self.eps)

            # u_ = tf.matmul(v_hat, W)
            # u_hat = self.l2normalize(u_)
            _u = self.l2normalize(K.dot(_v, W), eps=self.eps)

        return _u, _v

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.update_weights()
            output = self.layer(inputs)
            self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
            return output

        return self.layer(inputs)
    
    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        # u_hat = self.u
        # v_hat = self.v  # init v vector

        u_hat, v_hat = self.power_iteration(w_reshaped, self.u, self.power_iterations)
        # v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
        # # v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)
        # v_hat = self.l2normalize(v_, self.eps)

        # u_ = tf.matmul(v_hat, w_reshaped)
        # # u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)
        # u_hat = self.l2normalize(u_, self.eps)

        # sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        sigma=K.dot(K.dot(v_hat, w_reshaped), K.transpose(u_hat))
        self.u.assign(u_hat)
        # self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)