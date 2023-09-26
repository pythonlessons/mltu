import os
import numpy as np

class Audio:
    """ Audio object

    Attributes:
        audio (np.ndarray): Audio array
        sample_rate (int): Sample rate
        init_successful (bool): True if audio was successfully read
        library (object): Library used to read audio, tested only with librosa
    """
    init_successful = False
    augmented=False
    
    def __init__(
            self, 
            audioPath: str, 
            sample_rate: int=22050,
            library=None
        ) -> None:
        if library is None:
            raise ValueError("library must be provided. (e.g. librosa object)")

        if isinstance(audioPath, str):
            if not os.path.exists(audioPath):
                raise FileNotFoundError(f"Image {audioPath} not found.")

            self._audio, self.sample_rate = library.load(audioPath, sr=sample_rate)
            self.path = audioPath
            self.init_successful = True

        else:
            raise TypeError(f"audioPath must be path to audio file, not {type(audioPath)}")
        
    @property
    def audio(self) -> np.ndarray:
        return self._audio
    
    @audio.setter
    def audio(self, value: np.ndarray):
        self.augmented = True
        self._audio = value

    @property
    def shape(self) -> tuple:
        return self._audio.shape
    
    def numpy(self) -> np.ndarray:
        return self._audio
    
    def __add__(self, other: np.ndarray) -> np.ndarray:
        self._audio = self._audio + other
        self.augmented = True
        return self
    
    def __len__(self) -> int:
        return len(self._audio)
    
    def __call__(self) -> np.ndarray:
        return self._audio
    
    def __repr__(self):
        return repr(self._audio)
    
    def __array__(self):
        return self._audio