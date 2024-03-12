import cv2
import typing
import numpy as np
from enum import Enum


class BboxType(Enum):
    XYWH = 1 # x center, y center, width, height
    XYXY = 2 # left, top, right, bottom
    LTWH = 3 # left, top, width, height


class Detection:
    """ Object to hold the information of a detection for simple manipulation and visualization
    """
    def __init__(
            self, 
            bbox, 
            label: str, 
            labels: dict={},
            bbox_type: BboxType=BboxType.XYWH,
            confidence: float=0.0, 
            image_path: str="", 
            width: int=None, 
            height: int=None,
            relative: bool=False
        ):
        """
        Args:
            bbox (list or np.ndarray): Bounding box coordinates
            label (str): Label of the detection
            labels (dict, optional): Dictionary of labels. Defaults to {}.
            bbox_type (BboxType, optional): Type of the bounding box coordinates. Defaults to BboxType.XYWH.
            confidence (float, optional): Confidence score of the detection. Defaults to 0.0.
            image_path (str, optional): Path to the image. Defaults to "".
            width (int, optional): Width of the image. Defaults to None.
            height (int, optional): Height of the image. Defaults to None.
            relative (bool, optional): Whether the bounding box coordinates are relative to the image size. Defaults to False.
        """
        self.bbox = np.array(bbox)
        self.label = label
        self.labels = labels
        self.bbox_type = bbox_type
        self.confidence = confidence
        self.image_path = image_path
        self.width = width
        self.height = height
        self.relative = relative

        self.augmented = False

        self._xywh = None
        self._xyxy = None

        self.validate()

    @property
    def labelId(self) -> int:
        return self.label2id(self.label)

    def label2id(self, label: str) -> int:
        labelId = {v: k for k, v in self.labels.items()}.get(label, None)
        if labelId is None:
            raise ValueError(f"label {label} not found in labels")
        
        return labelId
    
    @property
    def xywh(self):
        return self._xywh
    
    @xywh.setter
    def xywh(self, xywh: np.ndarray):
        if (xywh[:2] + xywh[2:] / 2 > 1).any():
            # fix the bbox to be in range [0, 1]
            self._xywh = self.xyxy2xywh(self.xywh2xyxy(xywh))
        else:
            self._xywh = xywh.clip(0, 1)

        self._xyxy = self.xywh2xyxy(self._xywh)

    @property
    def xyxy(self):
        return self._xyxy

    @staticmethod
    def xywh2xyxy(xywh: np.ndarray):
        """ Convert bounding box from x, y, width, height to x1, y1, x2, y2
        """
        x, y, w, h = xywh
        x, y = x - w / 2, y - h / 2
        return np.array([x, y, x+w, y+h]).clip(0, 1)
    
    @staticmethod
    def xyxy2xywh(xyxy: np.ndarray):
        """ Convert bounding box from x1, y1, x2, y2 to x, y, width, height
        """
        x, y, x2, y2 = xyxy
        w, h = x2 - x, y2 - y
        return np.array([x + w / 2, y + h / 2, w, h]).clip(0, 1)
    
    @staticmethod
    def ltwh2xywh(ltwh: np.ndarray):
        """ Convert bounding box from left, top, width, height to x, y, width, height
        """
        l, t, w, h = ltwh
        return np.array([l + w / 2, t + h / 2, w, h]).clip(0, 1)
    
    def validate(self):
        """ Validate the bounding box coordinates
        """
        assert self.bbox_type in BboxType, f"bbox_type must be one of {BboxType}"
        if not self.relative:
            if self.width is None or self.height is None:
                raise ValueError("width and height must be provided when relative is False")
            
            if (np.array(self.bbox) > 1.0).any():
                raise ValueError("bbox coordinates must be in range [0, 1] when relative is False")
            
            bbox = np.array(self.bbox) / np.array([self.width, self.height, self.width, self.height])

        else:
            bbox = self.bbox

        if self.bbox_type.name == "XYWH":
            self.xywh = bbox

        elif self.bbox_type.name == "XYXY":
            self.xywh = self.xyxy2xywh(bbox)

        elif self.bbox_type.name == "LTWH":
            self.xywh = self.ltwh2xywh(bbox)

        else:
            raise ValueError(f"bbox_type {self.bbox_type} not supported")
        
    def flip(self, direction: int):
        new_xywh = self.xywh
        if direction == 0: # mirror
            new_xywh[0] = 1 - new_xywh[0]

        elif direction == 1: # vertical
            new_xywh[1] = 1 - new_xywh[1]

        self.xywh = new_xywh
        
        self.augmented = True

    def dot(self, rotMat: np.ndarray, width: int, height: int):
        """ Apply transformation matrix to detection

        Args:
            matrix (np.ndarray): Transformation matrix
            width (int): Width of the image
            height (int): Height of the image

        Returns:
            Object with transformed coordinates
        """
        # get the four corners of the bounding box
        bb = np.array(self.xyxy) * np.array([self.width, self.height, self.width, self.height])
        bb = np.array(((bb[0],bb[1]),(bb[2],bb[1]),(bb[2],bb[3]),(bb[0],bb[3])))

        bb_rotated = np.vstack((bb.T, np.array((1,1,1,1)))) #Convert the array to [x,y,1] format to dot it with the rotMat
        bb_rotated = np.dot(rotMat, bb_rotated).T #Perform Dot product and get back the points in shape of (4,2)

        # get the new coordinates of the bounding box
        x_min = min(bb_rotated[:, 0])
        y_min = min(bb_rotated[:, 1])
        x_max = max(bb_rotated[:, 0])
        y_max = max(bb_rotated[:, 1])

        new_x = (x_min + x_max) / 2
        new_y = (y_min + y_max) / 2
        new_w = x_max - x_min
        new_h = y_max - y_min

        # Normalize to the new width and height
        new_x /= width
        new_y /= height
        new_w /= width
        new_h /= height

        self.xywh = np.array([new_x, new_y, new_w, new_h])

        self.width = width
        self.height = height
        self.augmented = True
        
        return self

    def applyToFrame(self, frame: np.ndarray, color: tuple=(0, 255, 0), thickness: int=2, **kwargs) -> np.ndarray:
        """ Draw the bounding box on the image
        """
        # Get the coordinates of the bounding box
        x, y, x2, y2 = (self.xyxy * np.array([self.width, self.height, self.width, self.height])).astype(np.int32)

        # Draw the bounding box on the image
        frame = cv2.rectangle(frame.copy(), (x, y), (x2, y2), color, thickness, **kwargs)

        label = f"{self.label}: {self.confidence:.2f}" if self.confidence > 0 else self.label

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_y = y - 10 if y - 10 > label_height else y + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            frame, (x, label_y - label_height), (x + label_width, label_y + label_height), color, cv2.FILLED
        )

        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def json(self):
        return {
            "xywh": self.xywh.tolist(),
            "label": self.label,
            "confidence": self.confidence,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height
        }
    

class Detections:
    """ Object to hold the information of multiple detections for simple manipulation and visualization"""
    def __init__(
            self, 
            labels: dict,
            width: int,
            height: int,
            detections: typing.Iterable[Detection] = [],
            image_path: str = "",
            color_palette: list=[]
        ) -> None:
        """ Initialize the Detections object

        Args:
            labels (dict): Dictionary of labels
            width (int): Width of the image
            height (int): Height of the image
            detections (typing.Iterable[Detection], optional): List of detections. Defaults to [].
            image_path (str, optional): Path to the image. Defaults to "".
            color_palette (list, optional): List of colors to use for the bounding boxes. Defaults to [].
        """
        self.labels = labels
        self.width = width
        self.height = height
        self.detections = detections
        self.image_path = image_path
        self.color_palette = color_palette

        self.validate()

    def label2id(self, label: str) -> int:
        labelId = {v: k for k, v in self.labels.items()}.get(label, None)
        if labelId is None:
            raise ValueError(f"label {label} not found in labels")
        
        return labelId

    def validate(self):
        for detection in self.detections:
            if not isinstance(detection, Detection):
                raise TypeError(f"detections must be iterable of Detection, not {type(detection)}")
            
            detection.width = self.width
            detection.height = self.height
            detection.labels = self.labels
            detection.image_path = self.image_path

        if isinstance(self.labels, list):
            self.labels = {i: label for i, label in enumerate(self.labels)}

    def applyToFrame(self, image: np.ndarray, **kwargs: dict) -> np.ndarray:
        """ Draw the detections on the image """
        for detection in self.detections:
            color = self.color_palette[detection.labelId] if len(self.color_palette) == len(self.labels) else (0, 255, 0)
            image = detection.applyToFrame(image, color=color, **kwargs)
        
        return image

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index: int):
        return self.detections[index]