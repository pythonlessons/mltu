import tensorflow as tf
from keras import layers
from transformer import TransformerLayer

def Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, 
                dropout_rate=0.1, encoder_input_size=None, decoder_input_size=None):
    inputs = [
        layers.Input(shape=(encoder_input_size,), dtype=tf.int64),
        layers.Input(shape=(decoder_input_size,), dtype=tf.int64),
    ]

    transformer = TransformerLayer(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            input_vocab_size=input_vocab_size,
                            target_vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate)(inputs)
    
    outputs = layers.Dense(target_vocab_size)(transformer)

    return tf.keras.Model(inputs=inputs, outputs=outputs)