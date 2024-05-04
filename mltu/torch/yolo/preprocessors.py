import torch
import typing
import numpy as np

class YoloPreprocessor:
    def __init__(self, device: torch.device, imgsz: int=640):
        self.device = device
        self.imgsz = imgsz

    def __call__(self, images, annotations) -> typing.Tuple[np.ndarray, dict]:
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

        batch["cls"] = torch.tensor(np.array(batch["cls"])).to(self.device)
        batch["bboxes"] = torch.tensor(np.array(batch["bboxes"])).to(self.device)
        batch["batch_idx"] = torch.tensor(np.array(batch["batch_idx"])).to(self.device)

        return np.array(images), batch