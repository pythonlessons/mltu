import cv2
import typing
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Transformer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: np.ndarray, label: np.ndarray, *args, **kwargs):
        raise NotImplementedError

class ExpandDims(Transformer):
    def __init__(self, axis: int=-1):
        self.axis = axis

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label

class ImageResizer(Transformer):
    """Resize image to (width, height)
    
    Attributes:
        width (int): Width of image
        height (int): Height of image
        keep_aspect_ratio (bool): Whether to keep aspect ratio of image
        padding_color (typing.Tuple[int]): Color to pad image
    """
    def __init__(
        self, 
        width: int, 
        height: int, 
        keep_aspect_ratio: bool=False, 
        padding_color: typing.Tuple[int]=(0, 0, 0)
        ) -> None:
        self._width = width
        self._height = height
        self._keep_aspect_ratio = keep_aspect_ratio
        self._padding_color = padding_color

    @staticmethod
    def resize_maintaining_aspect_ratio(image: np.ndarray, width_target: int, height_target: int, padding_color: typing.Tuple[int]=(0, 0, 0)) -> np.ndarray:
        """ Resize image maintaining aspect ratio and pad with padding_color.

        Args:
            image (np.ndarray): Image to resize
            width_target (int): Target width
            height_target (int): Target height
            padding_color (typing.Tuple[int]): Color to pad image

        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)

        return new_image

    def __call__(self, data: np.ndarray, label: np.ndarray):
        # Maintains aspect ratio and resizes with padding.
        if self._keep_aspect_ratio:
            image = self.resize_maintaining_aspect_ratio(data, self._width, self._height, self._padding_color)

            if isinstance(label, np.ndarray):
                label = self.resize_maintaining_aspect_ratio(label, self._width, self._height, self._padding_color)

            return image, label

        # Resizes without maintaining aspect ratio.
        return cv2.resize(data, (self._width, self._height)), label

class LabelIndexer(Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class LabelPadding(Transformer):
    """Pad label to max_word_length
    
    Attributes:
        max_word_length (int): Maximum length of label
        padding_value (int): Value to pad
    """
    def __init__(
        self, 
        max_word_length: int, 
        padding_value: int
        ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.pad(label, (0, self.max_word_length - len(label)), 'constant', constant_values=self.padding_value)

class SpectrogramPadding(Transformer):
    """Pad spectrogram to max_spectrogram_length
    
    Attributes:
        max_spectrogram_length (int): Maximum length of spectrogram
        padding_value (int): Value to pad
    """
    def __init__(
        self, 
        max_spectrogram_length: int, 
        padding_value: int
        ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        padded_spectrogram = np.pad(spectrogram, ((self.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)), mode='constant', constant_values=self.padding_value)

        return padded_spectrogram, label

class ImageShowCV2(Transformer):
    """Show image for visual inspection

    Attributes:
        verbose (bool): Whether to log label
    """
    def __init__(self, verbose: bool=True) -> None:
        self.verbose = verbose

    def __call__(self, data: np.ndarray, label: np.ndarray):
        """ Show image for visual inspection

        Args:
            data (np.ndarray): Image data
            label (np.ndarray): Label data
        
        Returns:
            data (np.ndarray): Image data
            label (np.ndarray): Label data (unchanged)
        """
        if self.verbose:
            if isinstance(label, (str, int, float)):
                logger.info('Label: ', label)

        cv2.imshow('image', data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return data, label