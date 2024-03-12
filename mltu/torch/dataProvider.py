import os
import typing
import pandas as pd

from ..augmentors import Augmentor
from ..transformers import Transformer
from ..dataProvider import DataProvider as BaseDataProvider

import multiprocessing
import concurrent.futures
import threading
import queue


class ThreadExecutor:
    def __init__(self, target: typing.Callable, workers: int = os.cpu_count()) -> None:
        self.target = target
        self.workers = workers
        
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def __call__(self, data: typing.Any) -> typing.Any:
        results = self._executor.map(self.target, data)
        return results
    
    def __exit__(self):
        self._executor.shutdown()


class Worker:
    """ Worker class for multiprocessing """
    def __init__(self, target: typing.Callable, timeout: int=1) -> None:
        self.target = target
        self.timeout = timeout
        self.conn_sender, self.conn_receiver = multiprocessing.Pipe()
        self.worker = multiprocessing.Process(target=self.run_worker, args=(target, self.conn_receiver))
        self.worker.start()
        self.busy = False

    def run_worker(self, function, conn_receiver):
        while True:
            try:
                if conn_receiver.poll(self.timeout):
                    data = conn_receiver.recv()
                    if data is not None:
                        if data == "stop":
                            break
                        else:
                            result = function(data)
                            conn_receiver.send(result)
                            data = None
            except:
                pass

        conn_receiver.send("stop")

    def send(self, data):
        if self.busy:
            return None
        
        self.busy = True
        self.data = data
        self.conn_sender.send(data)
        return self

    def get(self):
        if self.conn_sender.poll(self.timeout):
            results = self.conn_sender.recv()
            self.busy = False
            return results
        else:
            self.busy = True
            self.conn_sender.send(self.data)
            return self
    
    def __exit__(self):
        while True:
            if self.busy:
                continue

            self.conn_sender.send("stop")
            stop = self.conn_sender.recv()
            if stop == "stop":
                self.worker.join()
                self.worker.terminate()
                break


class ProcessExecutor:
    def __init__(self, target: typing.Callable, workers: int = os.cpu_count()) -> None:
        self.target = target
        self.workers = workers
        self.busy = False

        self.mp_workers = [Worker(target) for _ in range(self.workers)]

    def __enter__(self):
        return self
    
    def __exit__(self):
        for worker in self.mp_workers:
            worker.__exit__()
 
    def __call__(self, data) -> typing.Any:
        self.busy = True
        results = [None for _ in range(len(data))]
        finished = [0 for _ in range(len(data))]
        while True:
            # send data to workers
            for index, data_batch in enumerate(data):
                for worker in self.mp_workers:
                    if worker.busy == False and results[index] is None:
                        results[index] = worker.send(data_batch)
                        break

            # receive data from workers
            for i, result in enumerate(results):
                if result is None:
                    continue

                if isinstance(result, Worker):
                    results[i] = result.get()

                if not isinstance(result, Worker):
                    finished[i] = 1
           
            if sum(finished) == len(data):
                break
        
        self.busy = False
        return results


class DataProvider(BaseDataProvider):
    """ DataProvider for PyTorch with multiprocessing and multithreading support.
    """
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
            workers: int = os.cpu_count(),
            use_multiprocessing: bool = False,
            max_queue_size: int = 5,
            **kwargs
        ):
        """ Standardised object for providing data to a model while training.

        Args:
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
            workers (int, optional): Number of workers to use for multiprocessing or multithreading. Defaults to os.cpu_count().
            use_multiprocessing (bool, optional): Whether to use multiprocessing or multithreading. Defaults to multithreading (False).
            max_queue_size (int, optional): Maximum size of the queue. Defaults to 5.
            numpy (bool, optional): Whether to convert data to numpy. Defaults to True.
        """
        super(DataProvider, self).__init__(dataset=dataset, data_preprocessors=data_preprocessors, batch_size=batch_size, 
                                           shuffle=shuffle, initial_epoch=initial_epoch, augmentors=augmentors, transformers=transformers, batch_postprocessors=batch_postprocessors,
                                           skip_validation=skip_validation, limit=limit, use_cache=use_cache, **kwargs)
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def start_executor(self) -> None:
        """ Start the executor for multiprocessing or multithreading"""

        if not hasattr(self, "_executor"):
            if self.use_multiprocessing:
                try:
                    self._executor = ProcessExecutor(self.process_data, self.workers)
                except:
                    self.use_multiprocessing = False
                    self.logger.error("Failed to start multiprocessing, switching to multithreading")
                    self._executor = ThreadExecutor(self.process_data, self.workers)
            else:
                self._executor = ThreadExecutor(self.process_data, self.workers)

        if not hasattr(self, "_sequenceHandler"):
            self._sequenceHandler = SequenceHandler(self.__getitem__, len(self), self.max_queue_size, self._shuffle)

    def __iter__(self):
        """ Yealds a batch of processed data by index until the end of the dataset """
        self.start_executor()
        for index in range(len(self)):
            results = self._sequenceHandler(index)
            yield results

        self.__exit__()

    def __exit__(self):
        """ Shutdown and remove the executors """
        self._executor.__exit__()
        del self._executor
        self._sequenceHandler.__exit__()
        del self._sequenceHandler


class SequenceHandler:
    """ SequenceHandler used to preprocess batches of data in parallel in the background."""
    def __init__(
            self, 
            function: typing.Callable, 
            max_len: int, 
            queue_size: int=5, 
            shuffle: bool=True
        ) -> None:
        self.function = function
        self.max_len = max_len
        self.queue_size = queue_size
        self.shuffle = shuffle

        self.data_queue, self.result_queue = queue.Queue(), queue.Queue()

        self.thread_workers = [threading.Thread(target=self.worker_function) for _ in range(self.queue_size)]
        for worker in self.thread_workers:
            worker.start()

        self.results_dict = {}

    def worker_function(self):
        while True:
            # Get data from the queue (or other source)
            data_index = self.data_queue.get()

            if data_index == "stop":
                self.result_queue.put("stop")
                break
            
            # Perform some processing on the data
            result = self.function(data_index)
            
            # Put the result in the result queue
            self.result_queue.put({data_index: result})

    def __exit__(self):
        for _ in self.thread_workers:
            self.data_queue.put("stop")
            stop = self.result_queue.get()
            if stop == "stop":
                # worker stopped
                pass 
            else:
                print("Something went wrong")

    def __call__(self, index: int):
        if index == 0:
            for _index in range(self.queue_size):
                if _index >= self.max_len:
                    break
                self.data_queue.put(_index)

        while True:
            # join results_dict with received results
            # check if queue is not empty
            try:
                worker_results = self.result_queue.get(timeout=10)
            except:
                # used for debugging
                continue

            if not worker_results:
                continue

            if self.shuffle: # ignore batch order by index (works faster)
                next_index = index + self.queue_size
                if next_index < self.max_len:
                    self.data_queue.put(next_index)
                return list(worker_results.values())[0]

            # return batch in order by index
            self.results_dict.update(worker_results)
            result = self.results_dict.get(index, None)
            if result is not None:
                next_index = index + self.queue_size
                if next_index < self.max_len:
                    self.data_queue.put(next_index)
                del self.results_dict[index]
                return result