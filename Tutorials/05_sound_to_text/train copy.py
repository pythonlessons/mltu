import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

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
from mltu.configs import BaseModelConfigs

metadata_path = "/home/rokbal/Downloads/bengaliai-speech/bengaliai-speech/train.csv"
train_mp3s_path = "/home/rokbal/Downloads/bengaliai-speech/bengaliai-speech/train_mp3s"

metadata_df = pd.read_csv(metadata_path, header=None)

# def download_and_unzip(url, extract_to="Datasets", chunk_size=1024*1024):
#     http_response = urlopen(url)

#     data = b""
#     iterations = http_response.length // chunk_size + 1
#     for _ in tqdm(range(iterations)):
#         data += http_response.read(chunk_size)

#     tarFile = tarfile.open(fileobj=BytesIO(data), mode="r|bz2")
#     tarFile.extractall(path=extract_to)
#     tarFile.close()


# dataset_path = os.path.join("Datasets", "LJSpeech-1.1")
# if not os.path.exists(dataset_path):
#     download_and_unzip("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", extract_to="Datasets")

# dataset_path = "Datasets/LJSpeech-1.1"
# metadata_path = dataset_path + "/metadata.csv"
# wavs_path = dataset_path + "/wavs/"

# # Read metadata file and parse it
# metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
# metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
# metadata_df = metadata_df[["file_name", "normalized_transcription"]]

# structure the dataset where each row is a list of [wav_file_path, sound transcription]

# metadata_values = metadata_df.values.tolist()
train_dataset, val_dataset = [], []
for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
    if index == 0:
        continue
    if row[2] == "train":
        mp3_file_path = f"{train_mp3s_path}/{row[0]}.mp3"
        train_dataset.append([mp3_file_path, row[1]])
    else:
        mp3_file_path = f"{train_mp3s_path}/{row[0]}.mp3"
        val_dataset.append([mp3_file_path, row[1]])

    # if index == 10000:
    #     break

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()
# configs = BaseModelConfigs.load("Models/05_sound_to_text/202308251219/configs.yaml")
# configs.batch_size = 64
# # configs.save()

from concurrent.futures import ThreadPoolExecutor

def process_file(file_label):
    file_path, label = file_label

    spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    
    return [spectrogram.shape[0], label]

# Assuming you have a list of file-label pairs in train_dataset
with ThreadPoolExecutor(max_workers=16) as executor:  # You can adjust the number of threads as needed
    results = list(tqdm(executor.map(process_file, train_dataset), total=len(train_dataset)))

max_text_length, max_spectrogram_length = 0, 0
vocab = []  # Initialize vocab list
for result in results:
    max_spectrogram_length = max(max_spectrogram_length, result[0])
    max_text_length = max(max_text_length, len(result[1]))
    for symbol in result[1]:
        if symbol not in vocab:
            vocab.append(symbol)

configs.vocab = vocab
configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
configs.input_shape = [max_spectrogram_length, 193]
configs.save()

# Create a data provider for the dataset
train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# for data in train_data_provider:
#     pass

# Create a data provider for the dataset
val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Split the dataset into training and validation sets
# train_data_provider, val_data_provider = data_provider.split(split = 0.9)

# Creating TensorFlow model architecture
model = train_model(
    input_dim = configs.input_shape,
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
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
# train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
# val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
