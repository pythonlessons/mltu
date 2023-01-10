import os
import copy
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from .augmentors import Augmentor
from .transformers import Transformer

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataProvider(tf.keras.utils.Sequence):
    """ Standardised object for providing data to a TensorFlow model while training.

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
    def __init__(
        self, 
        dataset: typing.Union[str, list, pd.DataFrame],
        data_preprocessors: typing.List[typing.Callable],
        batch_size: int = 4,
        shuffle: bool = True,
        initial_epoch: int = 1,
        augmentors: typing.List[Augmentor] = None,
        transformers: typing.List[Transformer] = None,
        skip_validation: bool = False,
        limit: int = None,
        ) -> None:
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
    def augmentors(self) -> typing.List[Augmentor]:
        """ Return augmentors """
        return self._augmentors

    @augmentors.setter
    def augmentors(self, augmentors: typing.List[Augmentor]):
        """ Decorator for adding augmentors to the DataProvider """
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                if self._augmentors is not None:
                    self._augmentors.append(augmentor)
                else:
                    self._augmentors = [augmentor]

            else:
                logger.warning(f"Augmentor {augmentor} is not an instance of Augmentor.")

        return self._augmentors

    @property
    def transformers(self) -> typing.List[Transformer]:
        """ Return transformers """
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        """ Decorator for adding transformers to the DataProvider """
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]

            else:
                logger.warning(f"Transformer {transformer} is not an instance of Transformer.")

        return self._transformers

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
        """ Split current data provider into training and validation data providers. 
        
        Args:
            split (float, optional): The split ratio. Defaults to 0.9.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            train_data_provider (tf.keras.utils.Sequence): The training data provider.
            val_data_provider (tf.keras.utils.Sequence): The validation data provider.
        """
        if shuffle:
            np.random.shuffle(self._dataset)
            
        train_data_provider, val_data_provider = copy.copy(self), copy.copy(self)
        train_data_provider._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data_provider._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data_provider, val_data_provider

    def to_csv(self, path: str, index: bool=False) -> None:
        """ Save the dataset to a csv file 

        Args:
            path (str): The path to save the csv file.
            index (bool, optional): Whether to save the index. Defaults to False.
        """
        df = pd.DataFrame(self._dataset)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        """ Returns a batch of annotations by batch index in the dataset

        Args:
            index (int): The index of the batch in 

        Returns:
            batch_annotations (list): A list of batch annotations
        """
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations

    def __getitem__(self, index: int):
        """ Returns a batch of data by batch index"""
        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for index, (data, annotation) in enumerate(dataset_batch):
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)
            
            # If data is None, remove it from the dataset
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