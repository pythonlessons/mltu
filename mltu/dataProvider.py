import os
import copy
import typing
# import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
# from scipy import signal
# from scipy.io import wavfile

from tensorflow.keras.preprocessing.sequence import pad_sequences

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataProvider(tf.keras.utils.Sequence):
    """ Standardised object for providing data to a TensorFlow model. """
    def __init__(
        self, 
        dataset: typing.Union[str, list, pd.DataFrame],
        data_preprocessors: typing.List[typing.Callable],
        batch_size: int = 4,
        shuffle: bool = True,
        initial_epoch: int = 1,
        augmentors: typing.List[typing.Callable] = None,
        transformers: typing.List[typing.Callable] = None,
        skip_validation: bool = False,
        limit: int = None,
        ) -> None:
        """
        Args:
            dataset (str, list, pd.DataFrame): Path to dataset, list of data or pandas dataframe of data.
            data_preprocessors (list): List of data preprocessors. (e.g. [read image, read audio, etc.])
            batch_size (int, optional): The number of samples to include in each batch. Defaults to 4.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            initial_epoch (int): The initial epoch. Defaults to 1.
            augmentors (list, optional): List of augmentor functions. Defaults to None.
            transformers (list, optional): List of transformer functions. Defaults to None.
            skip_validation (bool, optional): Whether to skip validation. Defaults to False.
            limit (int, optional): Limit the number of samples in the dataset. Defaults to None.
        """
        super().__init__()
        self._dataset = self.validate(dataset, skip_validation, limit)
        self._data_preprocessors = data_preprocessors
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = initial_epoch
        self._augmentors = augmentors
        self._transformers = transformers
        self._skip_validation = skip_validation
        self._limit = limit
        self._step = 0

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(len(self._dataset) / self._batch_size))

    @property
    def epoch(self) -> int:
        """ Return Current Epoch"""
        return self._epoch

    @property
    def step(self) -> int:
        """ Return Current Step"""
        return self._step

    def on_epoch_end(self):
        """ Shuffle training dataset and increment epoch counter at the end of each epoch. """
        self._epoch += 1
        if self._shuffle:
            np.random.shuffle(self._dataset)

    def validate_list_dataset(self, dataset: list, skip_validation: bool = False) -> list:
        """ Validate a list dataset """
        if skip_validation:
            logger.info("Skipping Dataset validation...")
            return dataset

        validated_data = [data for data in tqdm(dataset, desc="Validating Dataset") if os.path.exists(data[0])]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset.")

        return validated_data

    def validate(self, dataset: typing.Union[str, list, pd.DataFrame], skip_validation: bool, limit: int) -> list:
        """ Validate the dataset and return the dataset """

        if limit:
            logger.info(f"Limiting dataset to {limit} samples.")
            dataset = dataset[:limit]

        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return self.validate_list_dataset(dataset, skip_validation)
        elif isinstance(dataset, pd.DataFrame):
            return self.validate_list_dataset(dataset.values.tolist(), skip_validation)
        else:
            raise TypeError("Dataset must be a path, list or pandas dataframe.")

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """ Split the dataset into training and validation sets """
        if shuffle:
            np.random.shuffle(self._dataset)
            
        train_dataset, val_dataset = copy.copy(self), copy.copy(self)
        train_dataset._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_dataset._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_dataset, val_dataset

    def to_csv(self, path: str) -> None:
        """ Save the dataset to a csv file """
        df = pd.DataFrame(self._dataset)
        df.to_csv(path, index=False)

    def get_batch_annotations(self, index: int) -> typing.List:
        """ Returns a batch of annotations by index"""
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations

    def __getitem__(self, index: int):
        """ Returns a batch of data by index"""
        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for index, (data, annotation) in enumerate(dataset_batch):
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)
            
            if data is None:
                self._dataset.remove(dataset_batch[index])
                continue

            batch_data.append(data)
            batch_annotations.append(annotation)

        # Apply augmentors to batch
        if self._augmentors is not None:
            for augmentor in self._augmentors:
                batch_data, batch_annotations = zip(*[augmentor(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])

        # Apply transformers to batch
        if self._transformers is not None:
            for transformer in self._transformers:
                batch_data, batch_annotations = zip(*[transformer(data, annotation) for data, annotation in zip(batch_data, batch_annotations)])

        return np.array(batch_data), np.array(batch_annotations)

class SoundDataProvider(DataProvider):
    def __init__(
        self, 
        vocab: typing.List[str] = None,
        *args,
        **kwargs
        ) -> None:
        # Intherit all arguments from parent class
        # super().__init__(dataset)
        # TensorFlowDataProvider.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.vocab = vocab

        # Mapping characters to integers
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        # Mapping integers back to original characters
        self.num_to_char = tf.keras.layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )

        # An integer scalar Tensor. The window length in samples.
        self.frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        self.frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        self.fft_length = 384

    def __getitem__(self, index: int):
        """ Returns a batch of data by index"""
        batch_annotations = self.get_batch_annotations(index)

        data, labels = [], []
        # bzz =[]
        for file_path, label in batch_annotations:

            # x, sr = librosa.load(file_path, sr=44100)
            # X = librosa.stft(x)
            # Xdb = librosa.amplitude_to_db(abs(X))
            # bzz.append(Xdb)

            # sample_rate, samples = wavfile.read(file_path)
            # frequencies, times, _spectrogram = signal.spectrogram(samples, sample_rate)

            # 1. Read wav file
            file = tf.io.read_file(file_path)
            # 2. Decode the wav file
            audio, _ = tf.audio.decode_wav(file)
            audio = tf.squeeze(audio, axis=-1)
            # 3. Change type to float
            audio = tf.cast(audio, tf.float32)
            # 4. Get the spectrogram
            spectrogram = tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)
            # 5. We only need the magnitude, which can be derived by applying tf.abs
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.math.pow(spectrogram, 0.5)
            # 6. normalisation
            means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
            stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
            spectrogram = (spectrogram - means) / (stddevs + 1e-10)
            ###########################################
            ##  Process the label
            ##########################################
            # 7. Convert label to Lower case
            label = tf.strings.lower(label)
            # 8. Split the label
            label = tf.strings.unicode_split(label, input_encoding="UTF-8")
            # 9. Map the characters in label to numbers
            label = self.char_to_num(label)
            # 10. Return a dict as our model is expecting two inputs

            # final_labels = pad_sequences([label], maxlen=len(label), padding='post', value=len(self.vocab))[0]

            data.append(spectrogram.numpy())
            labels.append(label.numpy())

        padded_data = pad_sequences(data, maxlen=max([len(d) for d in data]), padding='post', value=0, dtype='float32')
        padded_labels = pad_sequences(labels, maxlen=max([len(l) for l in labels]), padding='post', value=len(self.vocab))

        if self._transformers:
            for transformer in self._transformers:
                padded_data, padded_labels = zip(*[transformer(data, label) for data, label in zip(padded_data, padded_labels)])

        return np.array(padded_data), np.array(padded_labels)