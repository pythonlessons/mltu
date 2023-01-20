import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.dataProvider import DataProvider
from mltu.preprocessors import WavReader
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.losses import CTCloss
from mltu.callbacks import Model2onnx, TrainLogger
from mltu.metrics import CERMetric, WERMetric

from model import train_model
from configs import ModelConfigs

import stow
import pandas as pd
from tqdm import tqdm

import stow
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO

def download_and_unzip(url, extract_to='Datasets', chunk_size=1024*1024):
    http_response = urlopen(url)

    data = b''
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    tarFile = tarfile.open(fileobj=BytesIO(data), mode='r|bz2')
    tarFile.extractall(path=extract_to)
    tarFile.close()

dataset_path = stow.join('Datasets', 'LJSpeech-1.1')
if not stow.exists(dataset_path):
    download_and_unzip('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2', extract_to='Datasets')

dataset_path = "Datasets/LJSpeech-1.1"
metadata_path = dataset_path + "/metadata.csv"
wavs_path = dataset_path + "/wavs/"

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]

# structure the dataset where each row is a list of [wav_file_path, sound transcription]
dataset = [[f"Datasets/LJSpeech-1.1/wavs/{file}.wav", label] for file, label in metadata_df.values.tolist()]

# Create a ModelConfigs object to store model configurations
configs = ModelConfigs()

max_text_length, max_spectrogram_length = 0, 0
for file_path, label in tqdm(dataset):
    spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    valid_label = [c for c in label if c in configs.vocab]
    max_text_length = max(max_text_length, len(valid_label))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    configs.input_shape = (max_spectrogram_length, spectrogram.shape[1])

configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
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
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split = 0.9)

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
earlystopper = EarlyStopping(monitor='val_CER', patience=30, verbose=1, mode='min')
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode='auto')
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
train_data_provider.to_csv(stow.join(configs.model_path, 'train.csv'))
val_data_provider.to_csv(stow.join(configs.model_path, 'val.csv'))

# https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-tensorflow/3-visualizations-transforms
# https://www.tensorflow.org/tutorials/audio/simple_audio






    # audio, sr = librosa.load(wav_path) 

    # # Resample the audio to a consistent sample rate (if needed)
    # audio = librosa.resample(audio, sr, sample_rate)

    # # Extract the Spectrograms
    # spectrograms = librosa.stft(audio)
    # spectrograms = np.abs(spectrograms)

    # # Log-scaling the spectrogram
    # spectrograms = librosa.amplitude_to_db(spectrograms)

    # # Normalize the spectrograms
    # spectrograms = (spectrograms - np.mean(spectrograms)) / np.std(spectrograms)

    # mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # samples, sample_rate = librosa.load(wav_path, sr = 16000)
    # samples = librosa.resample(samples, sample_rate, 8000)

# # Must download and extract datasets manually from https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database to Datasets\IAM_Sentences
# sentences_txt_path = stow.join('Datasets', 'IAM_Sentences', 'ascii', 'sentences.txt')
# sentences_folder_path = stow.join('Datasets', 'IAM_Sentences', 'sentences')

# dataset, vocab, max_len = [], set(), 0
# words = open(sentences_txt_path, "r").readlines()
# for line in tqdm(words):
#     if line.startswith("#"):
#         continue

#     line_split = line.split(" ")
#     if line_split[2] == "err":
#         continue

#     folder1 = line_split[0][:3]
#     folder2 = line_split[0][:8]
#     file_name = line_split[0] + ".png"
#     label = line_split[-1].rstrip('\n')

#     # recplace '|' with ' ' in label
#     label = label.replace('|', ' ')

#     rel_path = stow.join(sentences_folder_path, folder1, folder2, file_name)
#     if not stow.exists(rel_path):
#         continue

#     dataset.append([rel_path, label])
#     vocab.update(list(label))
#     max_len = max(max_len, len(label))

# Create a ModelConfigs object to store model configurations
# configs = ModelConfigs()

# # Save vocab and maximum text length to configs
# configs.vocab = "".join(vocab)
# configs.max_text_length = max_len
# configs.save()

# Create a data provider for the dataset
# data_provider = DataProvider(
#     dataset=dataset,
#     skip_validation=True,
#     batch_size=configs.batch_size,
#     data_preprocessors=[ImageReader()],
#     transformers=[
#         ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
#         LabelIndexer(configs.vocab),
#         LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
#         ],
# )

# # Compile the model and print summary
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate), 
#     loss=CTCloss(), 
#     metrics=[
#         CERMetric(vocabulary=configs.vocab),
#         WERMetric(vocabulary=configs.vocab)
#         ],
#     run_eagerly=False
# )
# model.summary(line_length=110)

# # Define callbacks
# earlystopper = EarlyStopping(monitor='val_CER', patience=20, verbose=1, mode='min')
# checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor='val_CER', verbose=1, save_best_only=True, mode='min')
# trainLogger = TrainLogger(configs.model_path)
# tb_callback = TensorBoard(f'{configs.model_path}/logs', update_freq=1)
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_CER', factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode='auto')
# model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# # Train the model
# model.fit(
#     train_data_provider,
#     validation_data=val_data_provider,
#     epochs=configs.train_epochs,
#     callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
#     workers=configs.train_workers
# )

# # Save training and validation datasets as csv files
# train_data_provider.to_csv(stow.join(configs.model_path, 'train.csv'))
# val_data_provider.to_csv(stow.join(configs.model_path, 'val.csv'))





# import numpy as np
# import glob
# import tensorflow as tf
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import LSTM, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed
# from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import layers
# import matplotlib.pyplot as plt
# words=['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
# block_length = 0.050#->500ms
# voice_max_length = int(1/block_length)#->2s
# print("voice_max_length:", voice_max_length)
# def audioToTensor(filepath):
#     audio_binary = tf.io.read_file(filepath)
#     audio, audioSR = tf.audio.decode_wav(audio_binary)
#     audioSR = tf.get_static_value(audioSR)
#     audio = tf.squeeze(audio, axis=-1)
#     audio_lenght = int(audioSR * block_length)#->16000*0.5=8000
#     frame_step = int(audioSR * 0.008)#16000*0.008=128
#     if len(audio)<audio_lenght*voice_max_length:
#         audio = tf.concat([np.zeros([audio_lenght*voice_max_length-len(audio)]), audio], 0)
#     else:
#         audio = audio[-(audio_lenght*voice_max_length):]
#     spectrogram = tf.signal.stft(audio, frame_length=1024, frame_step=frame_step)
#     spectrogram = (tf.math.log(tf.abs(tf.math.real(spectrogram)))/tf.math.log(tf.constant(10, dtype=tf.float32))*20)-60
#     spectrogram = tf.where(tf.math.is_nan(spectrogram), tf.zeros_like(spectrogram), spectrogram)
#     spectrogram = tf.where(tf.math.is_inf(spectrogram), tf.zeros_like(spectrogram), spectrogram)
#     voice_length, voice = 0, []
#     nb_part = len(audio)//audio_lenght
#     part_length = len(spectrogram)//nb_part
#     partsCount = len(range(0, len(spectrogram)-part_length, int(part_length/2)))
#     parts = np.zeros((partsCount, part_length, 513))
#     for i, p in enumerate(range(0, len(spectrogram)-part_length, int(part_length/2))):
#         part = spectrogram[p:p+part_length]
#         parts[i] = part
#     return parts
# max_data = 200
# wordToId, idToWord = {}, {}
# testParts = audioToTensor('mini_speech_commands/go/0a9f9af7_nohash_0.wav')
# print(testParts.shape)
# X_audio, Y_word = np.zeros((max_data*8, testParts.shape[0], testParts.shape[1], testParts.shape[2])), np.zeros((max_data*8, 8))

# files = {}
# for i, word in enumerate(words):
#     wordToId[word], idToWord[i] = i, word
#     files[word] = glob.glob('mini_speech_commands/'+word+'/*.wav')
# for nb in range(0, max_data):
#     for i, word in enumerate(words):
#         audio = audioToTensor(files[word][nb])
#         X_audio[len(files)*nb + i] = audio
#         Y_word[len(files)*nb + i] = np.array(to_categorical([i], num_classes=len(words))[0])

# X_audio_test, Y_word_test = X_audio[int(len(X_audio)*0.8):], Y_word[int(len(Y_word)*0.8):]
# X_audio, Y_word = X_audio[:int(len(X_audio)*0.8)], Y_word[:int(len(Y_word)*0.8)]
# print("X_audio.shape: ", X_audio.shape)
# print("Y_word.shape: ", Y_word.shape)
# print("X_audio_test.shape: ", X_audio_test.shape)
# print("Y_word_test.shape: ", Y_word_test.shape)
# latent_dim=32
# encoder_inputs = Input(shape=(testParts.shape[0], None, None, 1))
# preprocessing = TimeDistributed(preprocessing.Resizing(6, 129))(encoder_inputs)
# normalization = TimeDistributed(BatchNormalization())(preprocessing)
# conv2d = TimeDistributed(Conv2D(34, 3, activation='relu'))(normalization)
# conv2d = TimeDistributed(Conv2D(64, 3, activation='relu'))(conv2d)
# maxpool = TimeDistributed(MaxPooling2D())(conv2d)
# dropout = TimeDistributed(Dropout(0.25))(maxpool)
# flatten = TimeDistributed(Flatten())(dropout)
# encoder_lstm = LSTM(units=latent_dim)(flatten)
# decoder_dense = Dense(len(words), activation='softmax')(encoder_lstm)
# model = Model(encoder_inputs, decoder_dense)
# model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
# model.summary(line_length=200)
# tf.keras.utils.plot_model(model, to_file='model_word.png', show_shapes=True)
# batch_size = 32
# epochs = 50
# history=model.fit(X_audio, Y_word, shuffle=False, batch_size=batch_size, epochs=epochs, steps_per_epoch=len(X_audio)//batch_size, validation_data=(X_audio_test, Y_word_test))
# model.save_weights('model_word.h5')
# model.save("model_word")
# metrics = history.history
# plt.plot(history.epoch, metrics['loss'], metrics['acc'])
# plt.legend(['loss', 'acc'])
# plt.savefig("learning-word.png")
# plt.show()
# plt.close()
# score = model.evaluate(X_audio_test, Y_word_test, verbose = 0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# print("Test voice recognition")
# for test_path, test_string in [('mini_speech_commands/go/0a9f9af7_nohash_0.wav', 'go'), ('mini_speech_commands/right/0c2ca723_nohash_0.wav', 'right')]:
#     print("test_string: ", test_string)
#     test_audio = audioToTensor(test_path)
#     result = model.predict(np.array([test_audio]))
#     max = np.argmax(result)
#     print("decoded_sentence: ", result, max, idToWord[max])