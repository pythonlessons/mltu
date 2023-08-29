import cv2
import typing
import numpy as np

from . import Image

import logging


class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
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
    def unpad_maintaining_aspect_ratio(padded_image: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        height, width = padded_image.shape[:2]
        ratio = min(width / original_width, height / original_height)

        delta_w = width - int(original_width * ratio)
        delta_h = height - int(original_height * ratio)
        left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        unpaded_image = padded_image[top:height-bottom, left:width-right]

        original_image = cv2.resize(unpaded_image, (original_width, original_height))

        return original_image

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

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        # Maintains aspect ratio and resizes with padding.
        if self._keep_aspect_ratio:
            image_numpy = self.resize_maintaining_aspect_ratio(image.numpy(), self._width, self._height, self._padding_color)
            if isinstance(label, Image):
                label_numpy = self.resize_maintaining_aspect_ratio(label.numpy(), self._width, self._height, self._padding_color)
                label.update(label_numpy)
        else:   
            # Resizes without maintaining aspect ratio.
            image_numpy = cv2.resize(image.numpy(), (self._width, self._height))
            if isinstance(label, Image):
                label_numpy = cv2.resize(label.numpy(), (self._width, self._height))
                label.update(label_numpy)

        image.update(image_numpy)

        return image, label

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
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)


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
        padded_spectrogram = np.pad(spectrogram, ((0, self.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=self.padding_value)

        return padded_spectrogram, label


class ImageShowCV2(Transformer):
    """Show image for visual inspection
    """
    def __init__(
        self, 
        verbose: bool = True,
        log_level: int = logging.INFO,
        name: str = "Image"
        ) -> None:
        """
        Args:
            verbose (bool): Whether to log label
            log_level (int): Logging level (default: logging.INFO)
            name (str): Name of window to show image
        """
        super(ImageShowCV2, self).__init__(log_level=log_level)
        self.verbose = verbose
        self.name = name

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
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
                self.logger.info(f"Label: {label}")

        cv2.imshow(self.name, image.numpy())
        if isinstance(label, Image):
            cv2.imshow(self.name+"Label", label.numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image, label