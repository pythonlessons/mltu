import cv2
import typing
import logging
import importlib
import numpy as np

from . import Image
from mltu.annotations.audio import Audio
from mltu.annotations.detections import Detections

""" Implemented Transformers:
- ExpandDims - Expand dimension of data
- ImageResizer - Resize image to (width, height)
- LabelIndexer - Convert label to index by vocab
- LabelPadding - Pad label to max_word_length
- ImageNormalizer - Normalize image to float value, transpose axis if necessary and convert to numpy
- SpectrogramPadding - Pad spectrogram to max_spectrogram_length
- AudioToSpectrogram - Convert Audio to Spectrogram
- ImageShowCV2 - Show image for visual inspection
"""

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
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_word_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_on_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self.padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self.max_word_length]
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)


class ImageNormalizer:
    """ Normalize image to float value, transpose axis if necessary and convert to numpy
    """
    def __init__(self, transpose_axis: bool=False):
        """ Initialize ImageNormalizer

        Args:
            transpose_axis (bool): Whether to transpose axis. Default: False
        """
        self.transpose_axis = transpose_axis
    
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[np.ndarray, typing.Any]:
        """ Convert each Image to numpy, transpose axis ant normalize to float value
        """
        img = image.numpy() / 255.0

        if self.transpose_axis:
            img = img.transpose(2, 0, 1)
        
        return img, annotation


class SpectrogramPadding(Transformer):
    """Pad spectrogram to max_spectrogram_length
    
    Attributes:
        padding_value (int): Value to pad
        max_spectrogram_length (int): Maximum length of spectrogram. Must be specified if use_on_batch is False. Default: None
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_spectrogram_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_spectrogram_length is None:
            raise ValueError("max_spectrogram_length must be specified if use_on_batch is False")

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in spectrogram])
            padded_spectrograms = []
            for spec in spectrogram:
                padded_spectrogram = np.pad(spec, ((0, max_len - spec.shape[0]), (0,0)), mode="constant", constant_values=self.padding_value)
                padded_spectrograms.append(padded_spectrogram)

            padded_spectrograms = np.array(padded_spectrograms)
            label = np.array(label)

            return padded_spectrograms, label

        padded_spectrogram = np.pad(spectrogram, ((0, self.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=self.padding_value)

        return padded_spectrogram, label


class AudioPadding(Transformer):
    def __init__(self, max_audio_length: int, padding_value: int = 0, use_on_batch: bool = False, limit: bool = False):
        super(AudioPadding, self).__init__()
        self.max_audio_length = max_audio_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch
        self.limit = limit

    def __call__(self, audio: Audio, label: typing.Any):
        # batched padding
        if self.use_on_batch:
            max_len = max([len(a) for a in audio])
            padded_audios = []
            for a in audio:
                # limit audio if it exceed max_audio_length
                padded_audio = np.pad(a, (0, max_len - a.shape[0]), mode="constant", constant_values=self.padding_value)
                padded_audios.append(padded_audio)

            padded_audios = np.array(padded_audios)
            # limit audio if it exceed max_audio_length
            if self.limit:
                padded_audios = padded_audios[:, :self.max_audio_length]

            return padded_audios, label

        audio_numpy = audio.numpy()
        # limit audio if it exceed max_audio_length
        if self.limit:
            audio_numpy = audio_numpy[:self.max_audio_length]
        padded_audio = np.pad(audio_numpy, (0, self.max_audio_length - audio_numpy.shape[0]), mode="constant", constant_values=self.padding_value)

        audio.audio = padded_audio

        return audio, label

class AudioToSpectrogram(Transformer):
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
        log_level (int): Logging level (default: logging.INFO)
    """
    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            log_level: int = logging.INFO,
        ) -> None:
        super(AudioToSpectrogram, self).__init__(log_level=log_level)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def __call__(self, audio: Audio, label: typing.Any):
        """Compute the spectrogram of a WAV file

        Args:
            audio (Audio): Audio object
            label (Any): Label of audio

        Returns:
            np.ndarray: Spectrogram of audio
            label (Any): Label of audio
        """

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = self.librosa.stft(audio.numpy(), hop_length=self.frame_step, win_length=self.frame_length, n_fft=self.fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram, label


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

        if isinstance(label, Detections):
            for detection in label:
                img = detection.applyToFrame(np.asarray(image.numpy()))
                image.update(img)

        cv2.imshow(self.name, image.numpy())
        if isinstance(label, Image):
            cv2.imshow(self.name+"Label", label.numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image, label