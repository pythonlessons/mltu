import cv2
import typing
import numpy as np
import logging

from . import Image

""" Implemented augmentors:
- RandomBrightness
- RandomRotate
- RandomErodeDilate
- RandomSharpen
- RandomGaussianBlur
- RandomSaltAndPepper
"""


def randomness_decorator(func):
    """ Decorator for randomness """
    def wrapper(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Decorator for randomness and type checking

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation
        """
        # check if image is Image object
        if not isinstance(image, Image):
            self.logger.error(f"image must be Image object, not {type(image)}, skipping augmentor")
            return image, annotation

        if np.random.rand() > self._random_chance:
            return image, annotation

        # return result of function
        return func(self, image, annotation)

    return wrapper


class Augmentor:
    """ Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
        log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
    """
    def __init__(self, random_chance: float=0.5, log_level: int = logging.INFO) -> None:
        self._random_chance = random_chance
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        # do the augmentation here
        return image, annotation


class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness """
    def __init__(
        self, 
        random_chance: float = 0.5,
        delta: int = 100,
        log_level: int = logging.INFO,
        ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
        """
        super(RandomBrightness, self).__init__(random_chance, log_level)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly adjust image brightness

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        value = 1 + np.random.uniform(-self._delta, self._delta) / 255

        hsv = np.array(image.HSV(), dtype = np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value

        hsv = np.uint8(np.clip(hsv, 0, 255))

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(img)

        return image, annotation


class RandomRotate(Augmentor):
    """ Randomly rotate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        angle: typing.Union[int, typing.List]=30, 
        borderValue: typing.Tuple[int, int, int]=None,
        log_level: int = logging.INFO,
        ) -> None:
        """ Randomly rotate image 

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            angle (int, list): Integer value or list of integer values for image rotation
            borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        """
        super(RandomRotate, self).__init__(random_chance, log_level)

        self._angle = angle
        self._borderValue = borderValue

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly rotate image

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation
        """
        # check if angle is list of angles or signle angle value
        if isinstance(self._angle, list):
            angle = float(np.random.choice(self._angle))
        else:
            angle = float(np.random.uniform(-self._angle, self._angle))

        # generate random border color
        borderValue = np.random.randint(0, 255, 3) if self._borderValue is None else self._borderValue
        borderValue = [int(v) for v in borderValue]

        # grab the dimensions of the image and then determine the centre
        center_x, center_y = image.center

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((image.height * sin) + (image.width * cos))
        nH = int((image.height * cos) + (image.width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        # perform the actual rotation and return the image
        img = cv2.warpAffine(image.numpy(), M, (nW, nH), borderValue=borderValue)
        image.update(img)

        return image, annotation


class RandomErodeDilate(Augmentor):
    """ Randomly erode and dilate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        kernel_size: typing.Tuple[int, int]=(1, 1), 
        log_level: int = logging.INFO,
        ) -> None:
        """ Randomly erode and dilate image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        """
        super(RandomErodeDilate, self).__init__(random_chance, log_level)
        self._kernel_size = kernel_size

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly erode and dilate image

        Args:
            image (Image): Image to be eroded and dilated
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Eroded and dilated image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        kernel = np.ones(self._kernel_size, np.uint8)

        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), kernel, iterations=1)

        image.update(img)

        return image, annotation


class RandomSharpen(Augmentor):
    """ Randomly sharpen image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        alpha: float = 0.25,
        lightness_range: typing.Tuple = (0.75, 2.0),
        kernel: np.ndarray = None,
        kernel_anchor: np.ndarray = None,
        log_level: int = logging.INFO,
        ) -> None:
        """ Randomly sharpen image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            alpha (float): Float between 0.0 and 1.0 setting bounds for random probability
            lightness_range (tuple): Tuple of 2 floats, setting bounds for random lightness change
            kernel (np.ndarray): Numpy array of kernel for image convolution
            kernel_anchor (np.ndarray): Numpy array of kernel anchor for image convolution
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        """
        super(RandomSharpen, self).__init__(random_chance, log_level)

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self._kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly sharpen image

        Args:
            image (Image): Image to be sharpened
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Sharpened image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        lightness = np.random.uniform(*self._ligtness_range)
        alpha = np.random.uniform(*self._alpha_range)

        kernel = self._kernel_anchor  * (self._lightness_anchor + lightness) + self._kernel
        kernel -= self._kernel_anchor
        kernel = (1 - alpha) * self._kernel_anchor + alpha * kernel

        # Apply sharpening to each channel
        r, g, b = cv2.split(image.numpy())
        r_sharp = cv2.filter2D(r, -1, kernel)
        g_sharp = cv2.filter2D(g, -1, kernel)
        b_sharp = cv2.filter2D(b, -1, kernel)

        # Merge the sharpened channels back into the original image
        image.update(cv2.merge([r_sharp, g_sharp, b_sharp]))

        return image, annotation
    

class RandomGaussianBlur(Augmentor):
    """ Randomly erode and dilate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        sigma: typing.Union[int, float] = 0.5,
        ) -> None:
        """ Randomly erode and dilate image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            sigma (int, float): standard deviation of the Gaussian kernel
        """
        super(RandomGaussianBlur, self).__init__(random_chance, log_level)
        self.sigma = sigma

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly blurs an image with a Gaussian filter

        Args:
            image (Image): Image to be blurred
            annotation (typing.Any): Annotation to be blurred

        Returns:
            image (Image): Blurred image
            annotation (typing.Any): Blurred annotation if necessary
        """
        img = cv2.GaussianBlur(image.numpy(), (0, 0), self.sigma)

        image.update(img)

        return image, annotation
    

class RandomSaltAndPepper(Augmentor):
    """ Randomly add Salt and Pepper noise to image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        salt_vs_pepper: float = 0.5,
        amount: float = 0.1,
        ) -> None:
        """ Randomly add Salt and Pepper noise to image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            salt_vs_pepper (float): ratio of salt vs pepper. Defaults to 0.5.
            amount (float): proportion of the image to be salted and peppered. Defaults to 0.1.
        """
        super(RandomSaltAndPepper, self).__init__(random_chance, log_level)
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount
        
        assert 0 <= salt_vs_pepper <= 1.0, "salt_vs_pepper must be between 0.0 and 1.0"
        assert 0 <= amount <= 1.0, "amount must be between 0.0 and 1.0"

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly add salt and pepper noise to an image

        Args:
            image (Image): Image to be noised
            annotation (typing.Any): Annotation to be noised

        Returns:
            image (Image): Noised image
            annotation (typing.Any): Noised annotation if necessary
        """
        img = image.numpy()
        height, width, channels = img.shape

        # Salt mode
        num_salt = int(self.amount * height * width * self.salt_vs_pepper)
        row_coords = np.random.randint(0, height, size=num_salt)
        col_coords = np.random.randint(0, width, size=num_salt)
        img[row_coords, col_coords, :] = [255, 255, channels]

        # Pepper mode
        num_pepper = int(self.amount * height * width * (1.0 - self.salt_vs_pepper))
        row_coords = np.random.randint(0, height, size=num_pepper)
        col_coords = np.random.randint(0, width, size=num_pepper)
        img[row_coords, col_coords, :] = [0, 0, channels]

        image.update(img)

        return image, annotation