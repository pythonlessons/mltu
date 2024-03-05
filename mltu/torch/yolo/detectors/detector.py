import cv2
import numpy as np
from mltu.inferenceModel import FpsWrapper

class BaseDetector:
    """Base class for the detectors in the YOLO family"""
    @staticmethod
    def preprocess(image: np.ndarray, height: int, width: int):
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (width, height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    @staticmethod
    def postprocess(outputs: np.ndarray, x_factor: float, y_factor: float, confidence_threshold: float=0.5, iou_threshold: float=0.5):
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(outputs))

        # Extract all classes confidence scores
        conf_scores = np.amax(outputs[:, 4:], axis=1)

        # Get the data index of the detections with scores above the confidence threshold
        indexes = np.where(conf_scores >= confidence_threshold)[0]

        # Extract the confidence scores of the detections
        scores = conf_scores[indexes]

        # Extract the class IDs of the detections
        class_ids = np.argmax(outputs[indexes, 4:], axis=1)

        # Extract the bounding box coordinates from the outputs and transform them to the original image space
        boxes = outputs[indexes, :4] * np.array([x_factor, y_factor, x_factor, y_factor])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        return boxes[indices], scores[indices], class_ids[indices]

    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        ...
        
    @FpsWrapper
    def __call__(self, image: np.ndarray):
        results = self.predict(image)
        return results