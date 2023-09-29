import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
tf.keras.mixed_precision.set_global_policy('mixed_float16') # mixed precission training for faster training time

import os
import tarfile
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.preprocessors import WavReader

from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs


def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
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
    download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")

dataset_path = "Datasets/LJSpeech-1.1"
metadata_path = dataset_path + "/metadata.csv"
wavs_path = dataset_path + "/wavs/"

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]

# structure the dataset where each row is a list of [wav_file_path, sound transcription]
dataset = [[f"Datasets/LJSpeech-1.1/wavs/{file}.wav", label.lower()] for file, label in metadata_df.values.tolist()]

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        LabelIndexer(configs.vocab),
        ],
    batch_postprocessors=[
        SpectrogramPadding(padding_value=0, use_on_batch=True),
        LabelPadding(padding_value=len(configs.vocab), use_on_batch=True),
    ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (None, 193),
    output_dim = len(configs.vocab),
    dropout=0.5
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
        ],
    run_eagerly=False
)
model.summary(line_length=110)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers,
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))