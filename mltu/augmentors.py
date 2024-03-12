import cv2
import typing
import numpy as np
import importlib
import logging

from . import Image
from mltu.annotations.audio import Audio
from mltu.annotations.detections import Detections, Detection

""" 
Implemented image augmentors:
- RandomBrightness
- RandomRotate
- RandomErodeDilate
- RandomSharpen
- RandomGaussianBlur
- RandomSaltAndPepper
- RandomMirror
- RandomFlip
- RandomDropBlock
- RandomMosaic

Implemented audio augmentors:
- RandomAudioNoise
- RandomAudioPitchShift
- RandomAudioTimeStretch
"""

def randomness_decorator(func):
    """ Decorator for randomness """
    def wrapper(self, data: typing.Union[Image, Audio], annotation: typing.Any) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """ Decorator for randomness and type checking

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation
        """
        # check if image is Image object
        if not isinstance(data, (Image, Audio)):
            self.logger.error(f"data must be Image or Audio object, not {type(data)}, skipping augmentor")
            # TODO instead of error convert image into Image object
            # TODO instead of error convert audio into Audio object
            return data, annotation

        if np.random.rand() > self._random_chance:
            return data, annotation

        # return result of function
        return func(self, data, annotation)

    return wrapper


class Augmentor:
    """ Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
        log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
    """
    def __init__(self, random_chance: float=0.5, log_level: int = logging.INFO, augment_annotation: bool = False) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self._augment_annotation = augment_annotation

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    def augment(self, data: typing.Union[Image, Audio]):
        """ Augment data """
        raise NotImplementedError

    @randomness_decorator
    def __call__(self, data: typing.Union[Image, Audio], annotation: typing.Any) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """ Randomly add noise to audio

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation if necessary
        """
        data = self.augment(data)

        if self._augment_annotation and isinstance(annotation, np.ndarray):
            annotation = self.augment(annotation)

        return data, annotation


class RandomBrightness(Augmentor):
    """ Randomly adjust image brightness """
    def __init__(
        self, 
        random_chance: float = 0.5,
        delta: int = 100,
        log_level: int = logging.INFO,
        augment_annotation: bool = False
        ) -> None:
        """ Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool, optional): If True, the annotation will be adjusted as well. Defaults to False.
        """
        super(RandomBrightness, self).__init__(random_chance, log_level, augment_annotation)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def augment(self, image: Image, value: float) -> Image:
        """ Augment image brightness """
        hsv = np.array(image.HSV(), dtype = np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value

        hsv = np.uint8(np.clip(hsv, 0, 255))

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(img)

        return image

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

        image = self.augment(image, value)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation, value)

        return image, annotation


class RandomRotate(Augmentor):
    """ Randomly rotate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        angle: typing.Union[int, typing.List]=30, 
        borderValue: typing.Tuple[int, int, int]=None,
        log_level: int = logging.INFO,
        augment_annotation: bool = True
        ) -> None:
        """ Randomly rotate image 

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            angle (int, list): Integer value or list of integer values for image rotation
            borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): If True, the annotation will be adjusted as well. Defaults to True.
        """
        super(RandomRotate, self).__init__(random_chance, log_level, augment_annotation)

        self._angle = angle
        self._borderValue = borderValue

    @staticmethod
    def rotate_image(image: np.ndarray, angle: typing.Union[float, int], borderValue: tuple=(0,0,0), return_rotation_matrix: bool=False) -> np.ndarray:
        # grab the dimensions of the image and then determine the centre
        height, width = image.shape[:2]
        center_x, center_y = (width // 2, height // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        # perform the actual rotation and return the image
        img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)

        if return_rotation_matrix:
            return img, M
        
        return img

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
        # check if angle is list of angles or a single angle value
        if isinstance(self._angle, list):
            angle = float(np.random.choice(self._angle))
        else:
            angle = float(np.random.uniform(-self._angle, self._angle))

        # generate random border color
        borderValue = np.random.randint(0, 255, 3) if self._borderValue is None else self._borderValue
        borderValue = [int(v) for v in borderValue]

        img, rotMat = self.rotate_image(image.numpy(), angle, borderValue, return_rotation_matrix=True)

        if self._augment_annotation:
            if isinstance(annotation, Image):
                # perform the actual rotation and return the annotation image
                annotation_image = self.rotate_image(annotation.numpy(), angle, borderValue=(0, 0, 0))
                annotation.update(annotation_image)
            elif isinstance(annotation, Detections):
                height, width = img.shape[:2]
                for detection in annotation:
                    detection.dot(rotMat, width, height)

        image.update(img)

        return image, annotation


class RandomErodeDilate(Augmentor):
    """ Randomly erode and dilate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        kernel_size: typing.Tuple[int, int]=(1, 1), 
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly erode and dilate image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean value to determine if annotation should be adjusted. Defaults to False.
        """
        super(RandomErodeDilate, self).__init__(random_chance, log_level, augment_annotation)
        self._kernel_size = kernel_size
        self.kernel = np.ones(self._kernel_size, np.uint8)

    def augment(self, image: Image) -> Image:
        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), self.kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), self.kernel, iterations=1)

        image.update(img)

        return image

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
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

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
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly sharpen image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            alpha (float): Float between 0.0 and 1.0 setting bounds for random probability
            lightness_range (tuple): Tuple of 2 floats, setting bounds for random lightness change
            kernel (np.ndarray): Numpy array of kernel for image convolution
            kernel_anchor (np.ndarray): Numpy array of kernel anchor for image convolution
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean to determine if annotation should be augmented. Defaults to False.
        """
        super(RandomSharpen, self).__init__(random_chance, log_level, augment_annotation)

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self._kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
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

        return image

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
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation
    

class RandomGaussianBlur(Augmentor):
    """ Randomly erode and dilate image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        sigma: typing.Union[int, float] = 0.5,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly erode and dilate image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            sigma (int, float): standard deviation of the Gaussian kernel
        """
        super(RandomGaussianBlur, self).__init__(random_chance, log_level, augment_annotation)
        self.sigma = sigma

    def augment(self, image: Image) -> Image:
        img = cv2.GaussianBlur(image.numpy(), (0, 0), self.sigma)

        image.update(img)

        return image

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
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation
    

class RandomSaltAndPepper(Augmentor):
    """ Randomly add Salt and Pepper noise to image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        salt_vs_pepper: float = 0.5,
        amount: float = 0.1,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly add Salt and Pepper noise to image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            salt_vs_pepper (float): ratio of salt vs pepper. Defaults to 0.5.
            amount (float): proportion of the image to be salted and peppered. Defaults to 0.1.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        """
        super(RandomSaltAndPepper, self).__init__(random_chance, log_level, augment_annotation)
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount
        
        assert 0 <= salt_vs_pepper <= 1.0, "salt_vs_pepper must be between 0.0 and 1.0"
        assert 0 <= amount <= 1.0, "amount must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
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

        return image

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
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation
    

class RandomMirror(Augmentor):
    """ Randomly mirror image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly mirror image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        """
        super(RandomMirror, self).__init__(random_chance, log_level, augment_annotation)

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly mirror an image

        Args:
            image (Image): Image to be mirrored
            annotation (typing.Any): Annotation to be mirrored

        Returns:
            image (Image): Mirrored image
            annotation (typing.Any): Mirrored annotation if necessary
        """
        image = image.flip(0)
        if self._augment_annotation and isinstance(annotation, Image):
            annotation = annotation.flip(0)

        elif isinstance(annotation, Detections):
            for detection in annotation:
                detection.flip(0)

        return image, annotation
    

class RandomFlip(Augmentor):
    """ Randomly flip image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        ) -> None:
        """ Randomly mirror image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        """
        super(RandomFlip, self).__init__(random_chance, log_level, augment_annotation)

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly mirror an image

        Args:
            image (Image): Image to be flipped
            annotation (typing.Any): Annotation to be flipped

        Returns:
            image (Image): Flipped image
            annotation (typing.Any): Flipped annotation if necessary
        """
        image = image.flip(1)
        if self._augment_annotation and isinstance(annotation, Image):
            annotation = annotation.flip(1)

        elif isinstance(annotation, Detections):
            for detection in annotation:
                detection.flip(1)

        return image, annotation
    

class RandomDropBlock(Augmentor):
    """ Randomly drop block from image"""
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        block_size_percentage: float = 0.05,
        keep_prob: float = 0.7,
        ) -> None:
        """ Randomly drop block from image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
            block_size_percentage (float): drop block size percentage relative to image size. Defaults to 0.05.
            keep_prob (float): Probability of keeping the block. Defaults to 0.7.
        """
        super(RandomDropBlock, self).__init__(random_chance, log_level, augment_annotation)
        self.block_size_percentage = block_size_percentage
        self.keep_prob = keep_prob

    @staticmethod
    def dropblock(image, block_percent=0.05, keep_prob=0.7):
        height, width = image.shape[:2]
        block_size = int(min(height, width) * block_percent)
        mask = np.ones((height, width), dtype=bool)

        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                if np.random.rand() > keep_prob:
                    mask[i:i+block_size, j:j+block_size] = False

        dropped_image = image * mask[..., np.newaxis]
        return dropped_image

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Randomly drop block from image

        Args:
            image (Image): Image to be dropped
            annotation (typing.Any): Annotation to be dropped

        Returns:
            image (Image): Dropped image
            annotation (typing.Any): Dropped annotation if necessary
        """
        img = self.dropblock(image.numpy(), self.block_size_percentage, self.keep_prob)
        image.update(img)

        return image, annotation
    

class RandomMosaic(Augmentor):
    def __init__(
        self, 
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
        target_size: typing.Tuple[int, int] = None,
        ) -> None:
        """ Randomly merge 4 images into one mosaic image
        
        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
            target_size (tuple): Tuple of 2 integers, setting target size for mosaic image. Defaults to None.
        """
        super(RandomMosaic, self).__init__(random_chance, log_level, augment_annotation)
        self.target_size = target_size
        self.images = []
        self.annotations = []

    @randomness_decorator
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ R

        Args:
            image (Image): Image to be used for mosaic
            annotation (typing.Any): Annotation to be used for mosaic

        Returns:
            image (Image): Mosaic image
            annotation (typing.Any): Mosaic annotation if necessary
        """
        if not isinstance(annotation, Detections):
            self.logger.error(f"annotation must be Detections object, not {type(annotation)}, skipping augmentor")
            return image, annotation

        self.images.append(image.numpy())
        self.annotations.append(annotation)

        if len(self.images) >= 4:
            # merge images and annotations into one image and annotation
            if self.target_size is None:
                # pick smalles target size and resize all images to that size
                target_size = (min([img.shape[0] for img in self.images]), min([img.shape[1] for img in self.images]))
            else:
                target_size = self.target_size

            images = [cv2.resize(img, target_size) for img in self.images[:4]]
            detections = []
            new_img = np.concatenate([
                np.concatenate(images[:2], axis=1), 
                np.concatenate(images[2:4], axis=1)
            ], axis=0)
            
            height, width = new_img.shape[:2]
            for index, annotation in enumerate(self.annotations[:4]):
                if isinstance(annotation, Detections):
                    for detection in annotation:
                        xywh = np.array(detection.xywh) / 2

                        if index in [1, 3]:
                            xywh[0] = xywh[0] + 0.5

                        if index in [2, 3]:
                            xywh[1] = xywh[1] + 0.5

                        new_detection = Detection(
                            xywh, 
                            label=detection.label, 
                            labels=detection.labels,
                            confidence=detection.confidence, 
                            image_path=detection.image_path, 
                            width=width, 
                            height=height,
                            relative=True
                        )
                        detections.append(new_detection)

            new_detections = Detections(
                labels=annotation.labels,
                width=width,
                height=height,
                detections=detections
            )
                    
            image.update(new_img)

            self.images = self.images[4:]
            self.annotations = self.annotations[4:]

            return image, new_detections

        return image, annotation


class RandomAudioNoise(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_noise_ratio (float): Maximum noise ratio to be added to audio. Defaults to 0.1.
    """
    def __init__(
            self, 
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            max_noise_ratio: float = 0.1,
        ) -> None:
        super(RandomAudioNoise, self).__init__(random_chance, log_level, augment_annotation)
        self.max_noise_ratio = max_noise_ratio

    def augment(self, audio: Audio) -> Audio:
        noise = np.random.uniform(-1, 1, len(audio))
        noise_ratio = np.random.uniform(0, self.max_noise_ratio)
        audio_noisy = audio + noise_ratio * noise

        return audio_noisy
    

class RandomAudioPitchShift(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_n_steps (int): Maximum number of steps to shift audio. Defaults to 5.
    """
    def __init__(
            self, 
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            max_n_steps: int = 5,
        ) -> None:
        super(RandomAudioPitchShift, self).__init__(random_chance, log_level, augment_annotation)
        self.max_n_steps = max_n_steps

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def augment(self, audio: Audio) -> Audio:
        random_n_steps = np.random.randint(-self.max_n_steps, self.max_n_steps)
        # changing default res_type "kaiser_best" to "linear" for speed and memory efficiency
        shift_audio = self.librosa.effects.pitch_shift(
            audio.numpy(), sr=audio.sample_rate, n_steps=random_n_steps, res_type="linear"
            )
        audio.audio = shift_audio

        return audio
    

class RandomAudioTimeStretch(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        min_rate (float): Minimum rate to stretch audio. Defaults to 0.8.
        max_rate (float): Maximum rate to stretch audio. Defaults to 1.2.
    """
    def __init__(
            self, 
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            min_rate: float = 0.8,
            max_rate: float = 1.2
        ) -> None:
        super(RandomAudioTimeStretch, self).__init__(random_chance, log_level, augment_annotation)
        self.min_rate = min_rate
        self.max_rate = max_rate

        try:
            librosa.__version__
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def augment(self, audio: Audio) -> Audio:
        random_rate = np.random.uniform(self.min_rate, self.max_rate)
        stretch_audio = librosa.effects.time_stretch(audio.numpy(), rate=random_rate)
        audio.audio = stretch_audio

        return audio