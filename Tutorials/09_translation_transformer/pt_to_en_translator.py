import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import Model2onnx

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tokenizers import CustomTokenizer

# from transformer import Transformer, TransformerLayer
from model import Transformer
from configs import ModelConfigs

configs = ModelConfigs()

en_training_data_path = "Datasets/en-es/opus.en-es-train.en"
en_validation_data_path = "Datasets/en-es/opus.en-es-dev.en"
es_training_data_path = "Datasets/en-es/opus.en-es-train.es"
es_validation_data_path = "Datasets/en-es/opus.en-es-dev.es"

def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        en_train_dataset = f.read().split("\n")[:-1]
    return en_train_dataset

en_training_data = read_files(en_training_data_path)
en_validation_data = read_files(en_validation_data_path)
es_training_data = read_files(es_training_data_path)
es_validation_data = read_files(es_validation_data_path)

max_lenght = 500
train_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_training_data, en_training_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
val_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_validation_data, en_validation_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
es_training_data, en_training_data = zip(*train_dataset)
es_validation_data, en_validation_data = zip(*val_dataset)

# prepare portuguese tokenizer, this is the input language
tokenizer = CustomTokenizer()
tokenizer.fit_on_texts(es_training_data)
tokenizer.update(es_validation_data)

# prepare english tokenizer, this is the output language
detokenizer = CustomTokenizer()
detokenizer.fit_on_texts(en_training_data)
detokenizer.update(en_validation_data)


# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

# train_examples, val_examples = examples['train'], examples['validation']

# train_dataset = []
# for pt, en in train_examples:
#     train_dataset.append([pt.numpy().decode('utf-8'), en.numpy().decode('utf-8')])

# val_dataset = []
# for pt, en in val_examples:
#     val_dataset.append([pt.numpy().decode('utf-8'), en.numpy().decode('utf-8')])

# # prepare portuguese tokenizer
# tokenizer = CustomTokenizer()
# tokenizer.fit_on_texts([train_dataset[i][0] for i in range(len(train_dataset))])
# tokenizer.update([val_dataset[i][0] for i in range(len(val_dataset))])
# tokenizer.save(configs.model_path + "/pt_tokenizer.json")

# # prepare english tokenizer
# detokenizer = CustomTokenizer()
# detokenizer.fit_on_texts([train_dataset[i][1] for i in range(len(train_dataset))])
# detokenizer.update([val_dataset[i][1] for i in range(len(val_dataset))])
# detokenizer.save(configs.model_path + "/eng_tokenizer.json")


def preprocess_inputs(data_batch, label_batch):
    encoder_input = np.zeros((len(data_batch), tokenizer.max_length)).astype(np.int64)
    decoder_input = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)
    decoder_output = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)

    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    for index, (data, label) in enumerate(zip(data_batch_tokens, label_batch_tokens)):
        encoder_input[index][:len(data)] = data
        decoder_input[index][:len(label)-1] = label[:-1] # Drop the [END] tokens
        decoder_output[index][:len(label)-1] = label[1:] # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output

train_dataProvider = DataProvider(
    train_dataset, 
    batch_size=configs.batch_size, 
    shuffle=True,
    batch_postprocessors=[preprocess_inputs]
    )

# for data in train_dataProvider:
#     pass

val_dataProvider = DataProvider(
    val_dataset, 
    batch_size=configs.batch_size, 
    shuffle=True,
    batch_postprocessors=[preprocess_inputs]
    )

transformer = Transformer(
    num_layers=configs.num_layers,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    input_vocab_size=len(tokenizer)+1,
    target_vocab_size=len(detokenizer)+1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length
    )

transformer.summary()

# transformer(train_dataProvider[0][0], training=False)
# transformer.load_weights("test/model.h5")

# test = transformer(data[0], training=False)
# transformer.summary()


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, mask_value=0, reduction='none') -> None:
        super(MaskedLoss, self).__init__()
        self.mask_value = mask_value
        self.reduction = reduction
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = y_true != self.mask_value
        loss = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss
    
def masked_accuracy(y_true, y_pred):
    pred = tf.argmax(y_pred, axis=2)
    label = tf.cast(y_true, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

# vocabulary = tf.constant(eng_tokenizer.list())
# vocabulary = tf.constant(list(self.vocab))
# wer = WERMetric.get_wer(self.sen_true, self.sen_pred, vocabulary).numpy()

# @tf.function
# def wer(y_true, y_pred):
#     pred = tf.argmax(y_pred, axis=2)
#     label = tf.cast(y_true, pred.dtype)

#     wer = WERMetric.get_wer(pred, label, vocabulary, padding=0, separator=" ")

#     # pred_str = pt_tokenizer.detokenize(pred.numpy())
#     # label_str = eng_tokenizer.detokenize(label.numpy())
#     # wer = get_wer(pred_str, label_str)

#     return wer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(tf.cast(self.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(configs.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


transformer.compile(
    loss=MaskedLoss(),
    optimizer=optimizer,
    metrics=[masked_accuracy],
    run_eagerly=False
    )


# Define callbacks
earlystopper = EarlyStopping(monitor="val_masked_accuracy", patience=10, verbose=1, mode="max")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_masked_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=False)
tb_callback = TensorBoard(f"{configs.model_path}/logs")
reduceLROnPlat = ReduceLROnPlateau(monitor="val_masked_accuracy", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="max")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5", metadata={"tokenizer": tokenizer.dict(), "detokenizer": detokenizer.dict()}, save_on_epoch_end=True)


transformer.fit(
    train_dataProvider, 
    validation_data=val_dataProvider, 
    epochs=configs.train_epochs,
    callbacks=[
        checkpoint, 
        tb_callback, 
        reduceLROnPlat,
        model2onnx
        ]
    )