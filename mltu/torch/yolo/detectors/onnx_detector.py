import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.torch.yolo.detectors.detector import BaseDetector
from mltu.annotations.detections import BboxType, Detection, Detections

class Detector(OnnxInferenceModel, BaseDetector):
    """ YOLOv8 detector using onnxruntime"""
    def __init__(
            self, 
            model_path: str,
            input_width: int, 
            input_height: int, 
            confidence_threshold: float=0.5, 
            iou_threshold: float=0.5, 
            classes: dict = None, 
            return_raw_output: bool=False,
            *args, **kwargs
        ):
        """
        Args:
            model_path (str): Path to the model file
            input_width (int): Input width to use for the model
            input_height (int): Input height to use for the model
            confidence_threshold (float, optional): Confidence threshold for filtering the predictions. Defaults to 0.5.
            iou_threshold (float, optional): Intersection over union threshold for filtering the predictions. Defaults to 0.5.
            classes (dict, optional): Dictionary of class names. Defaults to None.
            return_raw_output (bool, optional): Return raw output of the model (return bounding boxes, scores, and class ids). Defaults to False.
        """
        super().__init__(model_path, *args, **kwargs)
        self.input_width = input_width
        self.input_height = input_height
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.return_raw_output = return_raw_output

        self.classes = classes or self.metadata.get("classes", None)
        if self.classes is None:
            raise ValueError("The classes must be provided")
        
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def predict(self, image: np.ndarray, **kwargs) -> Detections:
        img_height, img_width, _ = image.shape

        # Preprocess the image
        preprocessed_image = self.preprocess(image, self.input_height, self.input_width)
        
        # Perform inference on the preprocessed image
        preds = self.model.run(self.output_names, {self.input_names[0]: preprocessed_image})

        # Convert torch tensor to numpy array
        results = preds[0][0]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor, y_factor = img_width / self.input_width, img_height / self.input_height

        # Perform postprocessing on the predictions
        boxes, scores, class_ids = self.postprocess(results, x_factor, y_factor, self.confidence_threshold, self.iou_threshold)

        if self.return_raw_output:
            return boxes, scores, class_ids

        detections = []
        for bbox, conf, class_id in zip(boxes, scores, class_ids):
            detection = Detection(
                bbox = bbox,
                label = self.classes[class_id],
                labels = self.classes,
                bbox_type=BboxType.XYWH,
                confidence=conf,
                relative=False,
                width=img_width,
                height=img_height
            )
            detections.append(detection)

        return Detections(
            labels=self.classes,
            width=img_width, 
            height=img_height, 
            detections=detections, 
            color_palette=self.color_palette,
        )