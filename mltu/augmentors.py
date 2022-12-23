import cv2
import typing
import numpy as np

""" Implemented augmentors:
- RandomBrightness
- RandomRotate
- RandomErodeDilate
"""

class Augmentor:
    """ Object that should be inherited by all augmentors
    Args:
        image (np.ndarray): Image to augment
        annotation (np.ndarray): Annotation to augment

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Augmented image and mask
    """
    def __init__(self, random_chance: float=0.5) -> None:
        """
        Args:
            random_chance (float, optional): Chance of applying the augmentor. Defaults to 0.5.
        """
        self._random_chance = random_chance

    def __call__(self, image: np.ndarray, annotation: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        if np.random.random() <= self._random_chance:
            pass

        return image, annotation

class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness

    Args:
        image (np.ndarray): Image to be adjusted
        annotation (np.ndarray): Annotation to be adjusted

    Returns:
        image (np.ndarray): Adjusted image
        annotation (np.ndarray): Adjusted annotation
    """
    def __init__(self, random_chance:float=0.5, delta:int=100)->None:
        """ 
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability
            delta (int): Integer value for brightness adjustment
        """
        assert delta >= 0.0
        assert delta <= 255.0

        self._random_chance = random_chance
        self._delta = delta

    def __call__(self, image:np.ndarray, annotation:np.ndarray)->typing.Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() <= self._random_chance:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            value = 1 + np.random.uniform(-self._delta, self._delta) / 255

            hsv = np.array(image, dtype = np.float32)

            hsv[:, :, 1] = hsv[:, :, 1] * value
            hsv[:, :, 2] = hsv[:, :, 2] * value

            hsv = np.uint8(np.clip(hsv, 0, 255))

            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image, annotation

class RandomRotate(Augmentor):
    """ Randomly rotate image

    Args:
        image (np.ndarray): Image to be rotated
        annotation (np.ndarray): Annotation to be rotated

    Returns:
        image (np.ndarray): Rotated image
        annotation (np.ndarray): Rotated annotation
    """
    def __init__(self, random_chance:float=0.5, angle:int=10, borderValue:typing.Tuple[int, int, int]=(255, 255, 255))->None:
        """
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability
            angle (int): Integer value for rotation angle, in degrees
            borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
        """
        self._random_chance = random_chance
        self._angle = angle
        self._borderValue = borderValue

    def __call__(self, image:np.ndarray, annotation:np.ndarray)->typing.Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() <= self._random_chance:

            angle = np.random.uniform(-self._angle, self._angle)

            h, w, _ = image.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, m, (w, h), borderValue=self._borderValue)
            # Check if annotation is image mask
            if not isinstance(annotation, str):
                annotation = cv2.warpAffine(annotation, m, (w, h), borderValue=self._borderValue)

        return image, annotation

class RandomErodeDilate:
    """ Randomly erode and dilate image

    Args:
        image (np.ndarray): Image to be eroded and dilated

    Returns:
        image (np.ndarray): Eroded and dilated image
    """
    def __init__(self, random_chance:float=0.5, kernel_size:typing.Tuple[int, int]=(1, 1))->None:
        """
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability
            kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
        """
        self._random_chance = random_chance
        self._kernel_size = kernel_size

    def __call__(self, image:np.ndarray, annotation)->typing.Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() <= self._random_chance:

            kernel = np.ones(self._kernel_size, np.uint8)

            if np.random.rand() <= 0.5:
                image = cv2.erode(image, kernel, iterations=1)
            else:
                image = cv2.dilate(image, kernel, iterations=1)

        return image, annotation