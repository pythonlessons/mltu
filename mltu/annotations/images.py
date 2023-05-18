import os
import cv2
import typing

from abc import ABC
from abc import abstractmethod

import numpy as np

from PIL import Image as PilImage


class Image(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def center(self) -> tuple:
        pass

    @abstractmethod
    def RGB(self) -> np.ndarray:
        pass

    @abstractmethod
    def HSV(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, image: np.ndarray):
        pass

    @abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class CVImage(Image):
    """ Image class for storing image data and metadata (opencv based)

    Args:
        image (str or np.ndarray): Path to image or numpy.ndarray
        method (int, optional): OpenCV method for reading image. Defaults to cv2.IMREAD_COLOR.
        path (str, optional): Path to image. Defaults to "".
        color (str, optional): Color format of image. Defaults to "BGR".
    """

    init_successful = False

    def __init__(
            self,
            image: typing.Union[str, np.ndarray],
            method: int = cv2.IMREAD_COLOR,
            path: str = "",
            color: str = "BGR"
    ) -> None:
        super().__init__()
        
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self._image = cv2.imread(image, method)
            self.path = image
            self.color = "BGR"

        elif isinstance(image, np.ndarray):
            self._image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(f"Image must be either path to image or numpy.ndarray, not {type(image)}")

        self.method = method

        if self._image is None:
            return None

        self.init_successful = True

        # save width, height and channels
        self.width = self._image.shape[1]
        self.height = self._image.shape[0]
        self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        self._image = value

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self._image
        elif self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")
        
    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self._image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self._image = image

            # save width, height and channels
            self.width = self._image.shape[1]
            self.height = self._image.shape[0]
            self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

            return self

        else:
            raise TypeError(f"image must be numpy.ndarray, not {type(image)}")

    def flip(self, axis: int = 0):
        """ Flip image along x or y axis

        Args:
            axis (int, optional): Axis along which image will be flipped. Defaults to 0.

        Returns:
            Object with flipped points
        """
        # axis must be either 0 or 1
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        self._image = self._image[:, ::-1] if axis == 0 else self._image[::-1]

        return self

    def numpy(self) -> np.ndarray:
        return self._image
    
    def __call__(self) -> np.ndarray:
        return self._image


class PillowImage(Image):
    """ Image class for storing image data and metadata (pillow based)

    Args:
        image (str): Path to image
    """
    init_successful = False

    def __init__(
            self,
            image: str) -> None:
        super().__init__()

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self.path = image
            self._image = PilImage.open(image)

            self.init_successful = True
        else:
            raise TypeError("Image must be a path to an image")

        if self.is_animated:
            # initialize whatever attributes we can already determine at this stage, i.e. width & height.
            self.width = self._image.width
            self.height = self._image.height
            self.channels = None
        else:
            self._init_attributes()

    @property
    def is_animated(self) -> bool:
        return hasattr(self._image, "is_animated") and self._image.is_animated

    @property
    def image(self) -> np.ndarray:
        if self.is_animated:
            raise Exception("convert to single image first")

        return np.asarray(self._image)

    @image.setter
    def image(self, value: np.ndarray):
        self._image = PilImage.fromarray(value)

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self.image
        elif self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def _init_attributes(self):
        self.color = self._image.mode

        # save width, height and channels
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]

    def update(self, image: PilImage.Image):
        if isinstance(image, PilImage.Image):
            self._image = image
        elif isinstance(image, np.ndarray):
            self._image = PilImage.fromarray(image)
        else:
            raise TypeError(f"image must be a Pillow Image or np.ndarray, not {type(image)}")

        if not self.is_animated:
            self._init_attributes()

        return self

    def flip(self, axis: int = 0):
        """ Flip image along x or y axis

        Args:
            axis (int, optional): Axis along which image will be flipped. Defaults to 0.

        Returns:
            Object with flipped points
        """
        # axis must be either 0 or 1
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        if self.is_animated:
            raise Exception("convert to single image first")

        if axis == 0:
            self._image = PilImage.fromarray(np.asarray(self._image)[:, ::-1])
        else:
            self._image = PilImage.fromarray(np.asarray(self._image)[::-1])

        return self

    def numpy(self) -> np.ndarray:
        return self.image

    def __call__(self) -> np.ndarray:
        return self.image

    def pillow(self) -> PilImage.Image:
        return self._image
