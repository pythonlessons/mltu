import os
import typing
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging

from . import Image
from mltu.annotations.audio import Audio

""" Implemented Preprocessors:
- ImageReader - Read image from path and return image and label
- AudioReader - Read audio from path and return audio and label
- WavReader - Read wav file with librosa and return spectrogram and label
- ImageCropper - Crop image to (width, height)
"""

class ImageReader:
    """Read image from path and return image and label"""
    def __init__(self, image_class: Image, log_level: int = logging.INFO, ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class

    def __call__(self, image_path: typing.Union[str, np.ndarray], label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Read image from path and return image and label
        
        Args:
            image_path (typing.Union[str, np.ndarray]): Path to image or numpy array
            label (Any): Label of image

        Returns:
            Image: Image object
            Any: Label of image
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = self._image_class(image=image_path)

        if not image.init_successful:
            image = None
            self.logger.warning(f"Image {image_path} could not be read, returning None.")

        return image, label

def import_librosa(object) -> None:
    """Import librosa using importlib"""
    try:
        version = object.librosa.__version__
    except:
        version = "librosa version not found"
        try:
            object.librosa = importlib.import_module('librosa')
            print("librosa version:", object.librosa.__version__)
        except:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

class AudioReader:
    """ Read audio from path and return audio and label

    Attributes:
        sample_rate (int): Sample rate. Defaults to None.
        log_level (int): Log level. Defaults to logging.INFO.
    """
    def __init__(
            self, 
            sample_rate = None,
            log_level: int = logging.INFO, 
        ) -> None:
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def __call__(self, audio_path: str, label: typing.Any) -> typing.Tuple[np.ndarray, typing.Any]:
        """ Read audio from path and return audio and label
        
        Args:
            audio_path (str): Path to audio
            label (Any): Label of audio

        Returns:
            Audio: Audio object
            Any: Label of audio
        """
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio {audio_path} not found.")
        else:
            raise TypeError(f"Audio {audio_path} is not a string.")

        audio = Audio(audio_path, sample_rate=self.sample_rate, library=self.librosa)

        if not audio.init_successful:
            audio = None
            self.logger.warning(f"Audio {audio_path} could not be read, returning None.")

        return audio, label
    
class WavReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """

    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            *args, **kwargs
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        matplotlib.interactive(False)
        # import librosa using importlib
        import_librosa(self)
        
    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        import_librosa(WavReader)

        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path) 

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = WavReader.librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram

    @staticmethod
    def plot_raw_audio(wav_path: str, title: str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
            title (str, optional): Title
        """
        import_librosa(WavReader)
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path, sr=sr)

        duration = len(audio) / orig_sr

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title("Audio Plot")
        plt.ylabel("signal wave")
        plt.xlabel("time (s)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T
        
        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect="auto", origin="lower")
        plt.title(f"Spectrogram: {title}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length), label


class ImageCropper:
    """Crop image to (width, height)

    Attributes:
        width (int): Width of image
        height (int): Height of image
        wifth_offset (int): Offset for width
        height_offset (int): Offset for height
    """
    def __init__(
            self,
            width: int,
            height: int,
            width_offset: int = 0,
            height_offset: int = 0,
            log_level: int = logging.INFO
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self._width = width
        self._height = height
        self._width_offset = width_offset
        self._height_offset = height_offset

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        image_numpy = image.numpy()

        source_width, source_height = image_numpy.shape[:2][::-1]

        if source_width >= self._width:
            image_numpy = image_numpy[:, self._width_offset:self._width + self._width_offset]
        else:
            raise Exception("unexpected")

        if source_height >= self._height:
            image_numpy = image_numpy[self._height_offset:self._height + self._height_offset, :]
        else:
            raise Exception("unexpected")

        image.update(image_numpy)

        return image, label
