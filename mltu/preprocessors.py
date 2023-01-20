import cv2
import typing
import librosa
import numpy as np
import tensorflow as tf

class ImageReader:
    """Read image with cv2 from path and return image and label"""
    def __init__(self, method: int = cv2.IMREAD_COLOR, *args, **kwargs):
        self._method = method

    def __call__(self, image_path: str, label: str):
        return cv2.imread(image_path, self._method), label

class WavReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """
    def __init__(
        self, 
        frame_length: int=256, 
        frame_step: int=160,
        fft_length: int=384,
        *args, **kwargs
        ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

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
        # # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        # audio, orig_sr = librosa.load(wav_path) 

        # # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # # The resulting spectrogram is also transposed for convenience
        # spectrogram = librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

        # # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        # spectrogram = np.abs(spectrogram)

        # # Take the square root of the magnitude spectrum to obtain the log spectrogram
        # spectrogram = np.power(spectrogram, 0.5)

        # # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # # A small value of 1e-10 is added to the denominator to prevent division by zero.
        # spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        with tf.device('/cpu:0'):
            # 1. Read wav file
            file = tf.io.read_file(wav_path)
            # 2. Decode the wav file
            audio, _ = tf.audio.decode_wav(file)
            audio = tf.squeeze(audio, axis=-1)
            # 3. Change type to float
            audio = tf.cast(audio, tf.float32)
            # 4. Get the spectrogram
            spectrogram = tf.signal.stft(
                audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
            )
            # 5. We only need the magnitude, which can be derived by applying tf.abs
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.math.pow(spectrogram, 0.5)
            # 6. normalisation
            means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
            stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
            spectrogram = (spectrogram - means) / (stddevs + 1e-10)

        return spectrogram.numpy()

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