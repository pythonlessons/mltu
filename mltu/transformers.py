import cv2
import typing
import numpy as np

class Transformer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: np.ndarray, label: np.ndarray, *args, **kwargs):
        raise NotImplementedError

class ExpandDims(Transformer):
    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, -1), label

class ImageResizer(Transformer):
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return cv2.resize(data, (self._width, self._height)), label

class LabelIndexer(Transformer):
    def __init__(self, vocab: typing.List[str]):
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label])

class LabelPadding(Transformer):
    def __init__(self, max_word_length: int, padding_value: int):
        self.max_word_length = max_word_length
        self.padding_value = padding_value

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.pad(label, (0, self.max_word_length - len(label)), 'constant', constant_values=self.padding_value)