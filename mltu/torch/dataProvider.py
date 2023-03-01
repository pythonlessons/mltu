import os
import numpy as np
import concurrent.futures

from ..dataProvider import DataProvider as dataProvider

class DataProvider(dataProvider):
    def __init__(
            self, 
            *args, 
            workers: int = os.cpu_count(),
            use_multiprocessing: bool = False,
            **kwargs,
        ):
        super(DataProvider, self).__init__(*args, **kwargs)
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self._executor = None

    def start_executor(self) -> None:
        """ Start the executor for multiprocessing or multithreading"""
        if self.use_multiprocessing:
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=min(self._batch_size, self.workers))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(self._batch_size, self.workers))

    def __getitem__(self, index: int):
        """ Returns a batch of data by index"""

        dataset_batch = self.get_batch_annotations(index)

        if self._executor is None:
            self.start_executor()

        batch_data, batch_annotations = zip(*self._executor.map(self.process_data, dataset_batch))

        return np.array(batch_data), np.array(batch_annotations)