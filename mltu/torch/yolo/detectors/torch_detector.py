import torch
import numpy as np
from mltu.torch.yolo.detectors.detector import BaseDetector
from mltu.annotations.detections import BboxType, Detection, Detections

class Detector(BaseDetector):
    def __init__(
            self, 
            model,
            input_width: int, 
            input_height: int, 
            classes: dict, 
            confidence_threshold: float=0.5, 
            iou_threshold: float=0.5, 
            device: str="cuda"
        ):
        super().__init__()
        self.model = model
        self.input_width = input_width
        self.input_height = input_height
        self.classes = classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def predict(self, image: np.ndarray, **kwargs) -> Detections:
        img_height, img_width, _ = image.shape

        # Preprocess the image
        preprocessed_image = self.preprocess(image, self.input_height, self.input_width)
        
        # Perform inference on the preprocessed image
        preds = self.model(torch.tensor(preprocessed_image).to(self.device))

        # Convert torch tensor to numpy array
        results = preds[0].cpu().detach().numpy()

        # Calculate the scaling factors for the bounding box coordinates
        x_factor, y_factor = img_width / self.input_width, img_height / self.input_height

        # Perform postprocessing on the predictions
        boxes, scores, class_ids = self.postprocess(results, x_factor, y_factor, self.confidence_threshold, self.iou_threshold)

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