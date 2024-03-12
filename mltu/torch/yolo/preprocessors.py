import torch
import numpy as np

class YoloPreprocessor:
    def __init__(self, device, imgsz=640):
        self.device = device
        self.imgsz = imgsz

    def __call__(self, images, annotations):
        batch = {
            "ori_shape": [],
            "resized_shape": [],
            "cls": [],
            "bboxes": [],
            "batch_idx": [],
        }

        for i, (image, detections) in enumerate(zip(images, annotations)):
            batch["ori_shape"].append([detections.height, detections.width])
            batch["resized_shape"].append([self.imgsz, self.imgsz])
            for detection in detections:
                batch["cls"].append([detection.labelId])
                batch["bboxes"].append(detection.xywh)
                batch["batch_idx"].append(i)

        batch["cls"] = torch.tensor(batch["cls"]).to(self.device)
        batch["bboxes"] = torch.tensor(batch["bboxes"]).to(self.device)
        batch["batch_idx"] = torch.tensor(batch["batch_idx"]).to(self.device)

        return np.array(images), batch