import os
import cv2
import typing
import numpy as np

class Image:
    """ Image class for storing image data and metadata

    Args:
        image (str or np.ndarray): Path to image or numpy.ndarray
        method (int, optional): OpenCV method for reading image. Defaults to cv2.IMREAD_COLOR.
        path (str, optional): Path to image. Defaults to "".
        color (str, optional): Color format of image. Defaults to "".
    """
    def __init__(
        self,
        image: typing.Union[str, np.ndarray],
        method: int = cv2.IMREAD_COLOR,
        path: str = "",
        color: str = "BGR"
        ) -> None:
        
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self.image = cv2.imread(image, method)
            self.path = image
            self.color = "BGR"

        elif isinstance(image, np.ndarray):
            self.image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(f"Image must be either path to image or numpy.ndarray, not {type(image)}")

        self.method = method

        if self.image is None:
            return None

        # save width, height and channels
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @property
    def center(self) -> tuple:
        return (self.width // 2, self.height // 2)

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

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self.image = image

            # save width, height and channels
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
            self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]

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

        self.image = self.image[:, ::-1] if axis == 0 else self.image[::-1]

        return self

    def numpy(self) -> np.ndarray:
        return self.image
    
    def __call__(self) -> np.ndarray:
        return self.image