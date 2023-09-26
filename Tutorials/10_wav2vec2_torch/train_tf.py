import tensorflow as tf
try:
    [
        tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in tf.config.experimental.list_physical_devices("GPU")
    ]
except:
    pass

from keras import layers
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding

from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay

from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import pandas as pd

from configs import ModelConfigs

configs = ModelConfigs()
from transformers import TFWav2Vec2ForCTC
from mltu.preprocessors import AudioReader


train_dataset = pd.read_csv("Models/10_wav2vec2_torch/202309171434/train.csv").values.tolist()
validation_dataset = pd.read_csv("Models/10_wav2vec2_torch/202309171434/val.csv").values.tolist()

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=16000),
        ],
    transformers=[
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_label_length, padding_value=len(configs.vocab)),
        ],
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True)
    ],
    augmentors=[
        RandomAudioNoise(), 
        RandomAudioPitchShift(), 
        RandomAudioTimeStretch()
    ],
    use_cache=True,
)

test_dataProvider = DataProvider(
    dataset=validation_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=16000),
        ],
    transformers=[
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_label_length, padding_value=len(configs.vocab)),
        ],
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True)
    ],
    use_cache=True,
)

class CustomWav2Vec2Model(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)

        pretrained_name = "facebook/wav2vec2-base-960h"
        self.model = TFWav2Vec2ForCTC.from_pretrained(pretrained_name, vocab_size=output_dim, ignore_mismatched_sizes=True)
        self.model.freeze_feature_encoder() # https://huggingface.co/blog/fine-tune-wav2vec2-english

    def __call__(self, inputs):
        outputs = self.model(inputs)

        final_state = tf.nn.softmax(outputs.logits, axis=-1)

        return final_state

custom_model = tf.keras.Sequential([
    layers.Input(shape=(None,), name="input", dtype=tf.float32),
    CustomWav2Vec2Model(len(configs.vocab)+1)
])

for data in train_dataProvider:
    results = custom_model(data[0])
    break

custom_model.summary()
# configs.save()


# Compile the model and print summary
custom_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=configs.init_lr, weight_decay=configs.weight_decay), 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
)

# Define callbacks
warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    final_lr=configs.final_lr,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    initial_lr=configs.init_lr,
)
earlystopper = EarlyStopping(
    monitor="val_CER", patience=16, verbose=1, mode="min"
)
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.h5",
    monitor="val_CER",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=False,
)
tb_callback = TensorBoard(f"{configs.model_path}/logs")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5", metadata={"vocab": configs.vocab})

# Train the model
custom_model.fit(
    train_dataProvider,
    validation_data=test_dataProvider,
    epochs=configs.train_epochs,
    callbacks=[warmupCosineDecay, earlystopper, checkpoint, tb_callback, model2onnx],
    max_queue_size=configs.train_workers,
    workers=configs.train_workers,
    use_multiprocessing=True,
)