import tensorflow as tf
import keras
from keras import layers
from keras.models import Model

from mltu.model_utils import residual_block, activation_layer

import keras.backend as K
from keras import initializers, regularizers, constraints

# class Attention(keras.layers.Layer):
# Add attention layer to the deep learning network
class Attention(layers.Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        # context = K.sum(context, axis=1)
        return context


def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # expand dims to add channel dimension
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    # Convolution layer 1
    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    # Convolution layer 2
    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)
    
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # x = layers.Dropout(dropout)(x)

    # x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    attention = Attention()(x)
    x = layers.Multiply()([x, attention])

    # Dense layer
    x = layers.Dense(256)(x)
    x = activation_layer(x, activation='leaky_relu')
    x = layers.Dropout(dropout)(x)

    # Classification layer
    output = layers.Dense(output_dim + 1, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model