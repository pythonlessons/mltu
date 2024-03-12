import os
import copy
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm

from .augmentors import Augmentor
from .transformers import Transformer

import logging


class DataProvider:
    def __init__(
            self,
            dataset: typing.Union[str, list, pd.DataFrame],
            data_preprocessors: typing.List[typing.Callable] = None,
            batch_size: int = 4,
            shuffle: bool = True,
            initial_epoch: int = 1,
            augmentors: typing.List[Augmentor] = None,
            transformers: typing.List[Transformer] = None,
            batch_postprocessors: typing.List[typing.Callable] = None,
            skip_validation: bool = True,
            limit: int = None,
            use_cache: bool = False,
            log_level: int = logging.INFO,
            numpy: bool = True,
    ) -> None:
        """ Standardised object for providing data to a model while training.

        Attributes:
            dataset (str, list, pd.DataFrame): Path to dataset, list of data or pandas dataframe of data.
            data_preprocessors (list): List of data preprocessors. (e.g. [read image, read audio, etc.])
            batch_size (int): The number of samples to include in each batch. Defaults to 4.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            initial_epoch (int): The initial epoch. Defaults to 1.
            augmentors (list, optional): List of augmentor functions. Defaults to None.
            transformers (list, optional): List of transformer functions. Defaults to None.
            batch_postprocessors (list, optional): List of batch postprocessor functions. Defaults to None.
            skip_validation (bool, optional): Whether to skip validation. Defaults to True.
            limit (int, optional): Limit the number of samples in the dataset. Defaults to None.
            use_cache (bool, optional): Whether to cache the dataset. Defaults to False.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            numpy (bool, optional): Whether to convert data to numpy. Defaults to True.
        """
        self._dataset = dataset
        self._data_preprocessors = [] if data_preprocessors is None else data_preprocessors
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = initial_epoch
        self._augmentors = [] if augmentors is None else augmentors
        self._transformers = [] if transformers is None else transformers
        self._batch_postprocessors = [] if batch_postprocessors is None else batch_postprocessors
        self._skip_validation = skip_validation
        self._limit = limit
        self._use_cache = use_cache
        self._step = 0
        self._cache = {}
        self._on_epoch_end_remove = []
        self._numpy = numpy

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Validate dataset
        if not skip_validation:
            self._dataset = self.validate(dataset)
        else:
            self.logger.info("Skipping Dataset validation...")

        if limit:
            self.logger.info(f"Limiting dataset to {limit} samples.")
            self._dataset = self._dataset[:limit]

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
                self.logger.warning(f"Augmentor {augmentor} is not an instance of Augmentor.")

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
                self.logger.warning(f"Transformer {transformer} is not an instance of Transformer.")

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

        # Remove any samples that were marked for removal
        for remove in self._on_epoch_end_remove:
            self.logger.warning(f"Removing {remove} from dataset.")
            self._dataset.remove(remove)
        self._on_epoch_end_remove = []

    def validate_list_dataset(self, dataset: list) -> list:
        """ Validate a list dataset """
        validated_data = [data for data in tqdm(dataset, desc="Validating Dataset") if os.path.exists(data[0])]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset.")

        return validated_data

    def validate(self, dataset: typing.Union[str, list, pd.DataFrame]) -> typing.Union[list, str]:
        """ Validate the dataset and return the dataset """

        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return self.validate_list_dataset(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return self.validate_list_dataset(dataset.values.tolist())
        else:
            raise TypeError("Dataset must be a path, list or pandas dataframe.")

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[typing.Any, typing.Any]:
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
            
        train_data_provider, val_data_provider = copy.deepcopy(self), copy.deepcopy(self)
        train_data_provider._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data_provider._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data_provider, val_data_provider

    def to_csv(self, path: str, index: bool = False) -> None:
        """ Save the dataset to a csv file 

        Args:
            path (str): The path to save the csv file.
            index (bool, optional): Whether to save the index. Defaults to False.
        """
        df = pd.DataFrame(self._dataset)
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
    
    def start_executor(self) -> None:
        """ Start the executor to process data"""
        def executor(batch_data):
            for data in batch_data:
                yield self.process_data(data)

        if not hasattr(self, "_executor"):
            self._executor = executor

    def __iter__(self):
        """ Create a generator that iterate over the Sequence."""
        for index in range(len(self)):
            results = self[index]
            yield results

    def process_data(self, batch_data):
        """ Process data batch of data """
        if self._use_cache and batch_data[0] in self._cache and isinstance(batch_data[0], str):
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)
            
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self._on_epoch_end_remove.append(batch_data)
                return None, None
            
            if self._use_cache and batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        # Then augment, transform and postprocess the batch data
        for objects in [self._augmentors, self._transformers]:
            for _object in objects:
                data, annotation = _object(data, annotation)

        if self._numpy:
            try:
                data = data.numpy()
                annotation = annotation.numpy()
            except:
                pass

        return data, annotation

    def __getitem__(self, index: int):
        """ Returns a batch of processed data by index
        
        Args:
            index (int): index of batch
            
        Returns:
            tuple: batch of data and batch of annotations
        """
        if index==0:
            self.start_executor()

        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for data, annotation in self._executor(dataset_batch):
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue
            batch_data.append(data)
            batch_annotations.append(annotation)

        if self._batch_postprocessors:
            for batch_postprocessor in self._batch_postprocessors:
                batch_data, batch_annotations = batch_postprocessor(batch_data, batch_annotations)

            return batch_data, batch_annotations

        try:
            return np.array(batch_data), np.array(batch_annotations)
        except:
            return batch_data, batch_annotations