import cv2
import importlib
import numpy as np

class ImageReader:
    """Read image with cv2 from path and return image and label"""
    def __init__(self, method: int = cv2.IMREAD_COLOR, *args, **kwargs):
        self._method = method

    def __call__(self, image_path: str, label: str):
        return cv2.imread(image_path, self._method), label

class WavReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        sample_rate (int): Sample rate of the audio file in Hz, default 16000.
        frame_length (int): Frame length of the spectrogram, default 255.
        frame_step (int): Frame step of the spectrogram, default 128.
    """
    try:
        librosa = importlib.import_module('librosa')
    except ImportError:
        raise ImportError('librosa is not installed. Please install librosa first (pip install librosa).')

    def __init__(
        self, 
        sample_rate: int=16000, 
        frame_length: int=255, 
        frame_step: int=128,
        *args, **kwargs
        ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step

    @staticmethod
    def get_spectrogram(wav_path, sample_rate: int, frame_length: int, frame_step: int):

        # Read wav file
        audio, orig_sr = WavReader.librosa.load(wav_path) 

        # Resample the audio to a consistent sample rate (if needed)
        audio = WavReader.librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)

        # Get the spectrogram
        spectrogram = WavReader.librosa.stft(audio, hop_length=frame_step, win_length=frame_length)

        # We only need the magnitude, which can be derived by applying np.abs
        spectrogram = np.abs(spectrogram)

        # Log-scaling the spectrogram
        spectrogram = WavReader.librosa.amplitude_to_db(spectrogram)

        # Normalize the spectrograms
        spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

        return spectrogram

    def __call__(self, audio_path: str, label: str):
        """
        """
        return WavReader.get_spectrogram(audio_path, self.sample_rate, self.frame_length, self.frame_step), label