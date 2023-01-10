import cv2
import typing
import numpy as np

""" Implemented augmentors:
- RandomBrightness
- RandomRotate
- RandomErodeDilate
- RandomSharpen
"""

class Augmentor:
    """ Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
    """
    def __init__(self, random_chance: float=0.5) -> None:
        self._random_chance = random_chance

    def __call__(self, image: np.ndarray, annotation: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        if np.random.random() <= self._random_chance:
            pass

        return image, annotation

class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        delta (int): Integer value for brightness adjustment
    """
    def __init__(
        self, 
        delta: int = 100,
        *args, **kwargs
        ) -> None:
        super(RandomBrightness, self).__init__(*args, **kwargs)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def __call__(self, image: np.ndarray, annotation: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """ Randomly adjust image brightness

        Args:
            image (np.ndarray): Image to be adjusted
            annotation (np.ndarray): Mask to be adjusted

        Returns:
            image (np.ndarray): Adjusted image
            annotation (np.ndarray): Adjusted mask
        """
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

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        angle (int): Angle between 0 and 180, setting image rotation borders
        borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
        crop_borders (bool): Boolean value, setting if borders should be cropped after rotation
    """
    def __init__(
        self, 
        angle: int=30, 
        borderValue: typing.Tuple[int, int, int]=None,
        crop_borders: bool=False,
        *args, **kwargs
        ) -> None:
        super(RandomRotate, self).__init__(*args, **kwargs)

        self._angle = angle
        self._borderValue = borderValue
        self._crop_borders = crop_borders

    def __call__(self, image:np.ndarray, annotation:np.ndarray)->typing.Tuple[np.ndarray, np.ndarray]:
        """ Randomly rotate image

        Args:
            image (np.ndarray): Image to be rotated
            annotation (np.ndarray): Mask to be rotated

        Returns:
            image (np.ndarray): Rotated image
            annotation (np.ndarray): Rotated mask
        """
        if np.random.rand() <= self._random_chance:

            angle = np.random.uniform(-self._angle, self._angle)

            # generate random border color
            borderValue = np.random.randint(0, 255, 3) if self._borderValue is None else self._borderValue
            borderValue = [int(v) for v in borderValue]

            h, w, _ = image.shape
            if self._crop_borders:
                m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                image = cv2.warpAffine(image, m, (w, h), borderValue=borderValue)
                # Check if annotation is image mask
                if not isinstance(annotation, str):
                    annotation = cv2.warpAffine(annotation, m, (w, h), borderValue=0)

            else:                
                diagonal = round(np.sqrt((w*w) + (h*h)))
                top, bottom, left, right = round((diagonal-h) / 2), round((diagonal-h) / 2), round((diagonal-w) / 2), round((diagonal-w) / 2)
                padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value = borderValue)

                padded_height, padded_width = padded_image.shape[:2]

                transform_matrix = cv2.getRotationMatrix2D((padded_height/2, padded_width/2), angle, 1.0)

                rotated_image = cv2.warpAffine(padded_image, transform_matrix, (diagonal, diagonal), flags=cv2.INTER_LANCZOS4, borderValue=borderValue)

                # Find the indices of the non-black pixels in the image
                indices = np.argwhere(rotated_image != np.array(borderValue))

                # Find the minimum and maximum row and column indices of the non-black pixels
                min_row, max_row = indices[:, 0].min(), indices[:, 0].max()
                min_col, max_col = indices[:, 1].min(), indices[:, 1].max()

                # Crop the black borders from the image
                cropped_image = rotated_image[min_row:max_row+1, min_col:max_col+1]

                if not isinstance(annotation, str):

                    padded_annotation = cv2.copyMakeBorder(annotation, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

                    rotated_annotation = cv2.warpAffine(padded_annotation, transform_matrix, (diagonal, diagonal), flags=cv2.INTER_LANCZOS4, borderValue=0)

                    cropped_annotation = rotated_annotation[min_row:max_row+1, min_col:max_col+1]

                    return cropped_image, cropped_annotation

                return cropped_image, annotation

        return image, annotation

class RandomErodeDilate(Augmentor):
    """ Randomly erode and dilate image

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
    """
    def __init__(
        self, 
        kernel_size: typing.Tuple[int, int]=(1, 1), 
        *args, **kwargs
        ) -> None:
        super(RandomErodeDilate, self).__init__(*args, **kwargs)

        self._kernel_size = kernel_size

    def __call__(self, image:np.ndarray, annotation)->typing.Tuple[np.ndarray, np.ndarray]:
        """ Randomly erode and dilate image

        Args:
            image (np.ndarray): Image to be eroded and dilated

        Returns:
            image (np.ndarray): Eroded and dilated image
        """
        if np.random.rand() <= self._random_chance:

            kernel = np.ones(self._kernel_size, np.uint8)

            if np.random.rand() <= 0.5:
                image = cv2.erode(image, kernel, iterations=1)
            else:
                image = cv2.dilate(image, kernel, iterations=1)

        return image, annotation

class RandomSharpen(Augmentor):
    """ Randomly sharpen image

    Attributes:
        alpha (float): Float between 0.0 and 1.0 setting bounds for random probability
        lightness_range (tuple): Tuple of 2 floats, setting bounds for random lightness change
        kernel (np.ndarray): Numpy array of kernel for image convolution
        kernel_anchor (np.ndarray): Numpy array of kernel anchor for image convolution
        random_chance (float, optional): Chance of applying the augmentor, where 1.0 is always and 0.0 is never. Defaults to 0.5.
    """
    def __init__(
        self, 
        alpha: float = 0.25,
        lightness_range: typing.Tuple = (0.75, 2.0),
        kernel: np.ndarray = None,
        kernel_anchor: np.ndarray = None,
        *args, **kwargs
        ) -> None:
        super(RandomSharpen, self).__init__(*args, **kwargs)

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self._kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    def __call__(self, image: np.ndarray, annotation) -> typing.Tuple[np.ndarray, np.ndarray]:
        """ Randomly sharpen image

        Args:
            image (np.ndarray): Image to be sharpened

        Returns:
            image (np.ndarray): Sharpened image
        """
        if np.random.rand() <= self._random_chance:

            lightness = np.random.uniform(*self._ligtness_range)
            alpha = np.random.uniform(*self._alpha_range)

            kernel = self._kernel_anchor  * (self._lightness_anchor + lightness) + self._kernel
            kernel -= self._kernel_anchor
            kernel = (1 - alpha) * self._kernel_anchor + alpha * kernel

            # Apply sharpening to each channel
            r, g, b = cv2.split(image)
            r_sharp = cv2.filter2D(r, -1, kernel)
            g_sharp = cv2.filter2D(g, -1, kernel)
            b_sharp = cv2.filter2D(b, -1, kernel)

            # Merge the sharpened channels back into the original image
            image = cv2.merge([r_sharp, g_sharp, b_sharp])

        return image, annotation