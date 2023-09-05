import tensorflow as tf

try:
    [
        tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in tf.config.experimental.list_physical_devices("GPU")
    ]
except:
    pass

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import os
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

from concurrent.futures import ThreadPoolExecutor

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import SpectrogramPadding
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay

from mltu.preprocessors import WavReader
from mltu.tokenizers import CustomTokenizer

from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss, CERMetric
from mltu.tensorflow.transformer.callbacks import EncDecSplitCallback

from mltu.preprocessors import AudioReader
from mltu.transformers import AudioToSpectrogram
from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch


from model import SpeechTransformer
from configs import ModelConfigs


def download_and_unzip(url, extract_to="Datasets", chunk_size=1024 * 1024):
    http_response = urlopen(url)

    data = b""
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
    tarFile.extractall(path=extract_to)
    tarFile.close()


dataset_path = os.path.join("Datasets", "LJSpeech-1.1")
if not os.path.exists(dataset_path):
    download_and_unzip(
        "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        extract_to="Datasets",
    )

dataset_path = "Datasets/LJSpeech-1.1"
metadata_path = dataset_path + "/metadata.csv"
wavs_path = dataset_path + "/wavs/"

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
# metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
# metadata_df = metadata_df[["file_name", "normalized_transcription"]]

tokenizer = CustomTokenizer(char_level=True)
# label_list = metadata_df["normalized_transcription"].values.tolist()
# tokenizer.fit_on_texts(label_list)

# structure the dataset where each row is a list of [wav_file_path, sound transcription]
dataset = [
    [f"Datasets/LJSpeech-1.1/wavs/{file_name}.wav", normalized_transcription]
    for file_name, transcription, normalized_transcription in metadata_df.values.tolist()
]

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()
# tokenizer.save(configs.model_path + "/tokenizer.json")


def process_file(path_and_label):
    # audio, label = audioReader(*path_and_label)
    # mfcc, label = audioToMFCC(audio, label)
    # return mfcc.shape[0]
    spectrogram = WavReader.get_spectrogram(path_and_label[0], frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    return spectrogram.shape[0]

# # Assuming you have a list of file-label pairs in train_dataset
with ThreadPoolExecutor(max_workers=8) as executor:  # You can adjust the number of threads as needed
    results = list(tqdm(executor.map(process_file, dataset), total=len(dataset)))

configs.max_spectrogram_length = 2048 # max(results) 
configs.input_shape = [configs.max_spectrogram_length, 193]

tokenizer.fit_on_texts([label for _, label in dataset])


# max_text_length, max_spectrogram_length = 0, 0
# vocab = []  # Initialize vocab list
# for result in results:
#     max_spectrogram_length = max(max_spectrogram_length, result[0])
#     max_text_length = max(max_text_length, len(result[1]))
#     for symbol in result[1]:
#         if symbol not in vocab:
#             vocab.append(symbol)

# max_spectrogram_length = 0
# for file_path, label in tqdm(dataset):
#     spectrogram = WavReader.get_spectrogram(
#         file_path,
#         frame_length=configs.frame_length,
#         frame_step=configs.frame_step,
#         fft_length=configs.fft_length,
#     )
#     max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])


# configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]
# configs.max_spectrogram_length = max_spectrogram_length
# # configs.max_text_length = max_text_length
configs.save()

speech_transformer = SpeechTransformer(
    num_layers_encoder=configs.num_layers_encoder,
    num_layers_decoder=configs.num_layers_decoder,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    target_vocab_size=len(tokenizer) + 1,
    dropout_rate=0.1,
    encoder_input_shape=configs.input_shape,
    decoder_input_shape=[tokenizer.max_length],
    activation="gelu",
)
speech_transformer.summary()


# exit()


def preprocess_inputs(data_batch, label_batch):
    encoder_input = np.array(data_batch).astype(np.float32)
    decoder_input = np.zeros((len(label_batch), tokenizer.max_length)).astype(np.int64)
    decoder_output = np.zeros((len(label_batch), tokenizer.max_length)).astype(np.int64)

    label_batch_tokens = tokenizer.texts_to_sequences(label_batch)

    for index, label in enumerate(label_batch_tokens):
        decoder_input[index][: len(label) - 1] = label[:-1]  # Drop the [END] tokens
        decoder_output[index][: len(label) - 1] = label[1:]  # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output


# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader()
    ],
    transformers=[
        AudioToSpectrogram(
            frame_length=configs.frame_length,
            frame_step=configs.frame_step,
            fft_length=configs.fft_length,
        ),
        SpectrogramPadding(
            max_spectrogram_length=configs.max_spectrogram_length, padding_value=0
        ),
    ],
    batch_postprocessors=[preprocess_inputs],
    use_cache=True,
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

train_data_provider.augmentors = [
    RandomAudioNoise(),
    RandomAudioPitchShift(),
    RandomAudioTimeStretch(),
]

speech_transformer = SpeechTransformer(
    num_layers_encoder=configs.num_layers_encoder,
    num_layers_decoder=configs.num_layers_decoder,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    target_vocab_size=len(tokenizer) + 1,
    dropout_rate=0.1,
    encoder_input_shape=configs.input_shape,
    decoder_input_shape=[tokenizer.max_length],
)
speech_transformer.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=configs.init_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

speech_transformer.compile(
    loss=MaskedLoss(mask_value=0),
    optimizer=optimizer,
    metrics=[MaskedAccuracy(), CERMetric(end_token=tokenizer.end_token_index)],
    run_eagerly=False,
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
    monitor="val_masked_accuracy", patience=21, verbose=1, mode="max"
)
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.h5",
    monitor="val_masked_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
    save_weights_only=False,
)
tb_callback = TensorBoard(f"{configs.model_path}/logs")
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_masked_accuracy",
    factor=0.9,
    min_delta=1e-10,
    patience=10,
    verbose=1,
    mode="max",
)
model2onnx = Model2onnx(
    f"{configs.model_path}/model.h5",
    metadata={"tokenizer": tokenizer.dict()},
    save_on_epoch_end=False,
)
encDecSplitCallback = EncDecSplitCallback(configs.model_path, decoder_metadata={"tokenizer": tokenizer.dict()})

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

speech_transformer.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[
        warmupCosineDecay,
        earlystopper,
        checkpoint,
        tb_callback,
        reduceLROnPlat,
        model2onnx,
        encDecSplitCallback,
    ],
    max_queue_size=configs.train_workers,
    workers=configs.train_workers,
    use_multiprocessing=True,
)
