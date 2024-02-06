import os
import torch
from .utils.utils import yaml_load, guess_model_scale
from .models import DetectionModel
from .nn.utils import initialize_weights

class Yolo:
    def __init__(self, model: str):
        self.task = "detect"
        
        if model.endswith(".pt"):
            # load model from weights file
            self._load(model)
        elif model.endswith(".yaml"):
            # load model from config file
            self._new(model)
        else:
            raise ValueError("Model must be a .pt or .yaml file")
        
        pass

    def _load(self, model: str):
        # load model from weights file
        if os.path.exists(model):
            ckpt = torch.load(model, map_location=torch.device("cpu"))
            checkpoint = torch.load(model, map_location='cpu', weights_only=True)
            model = torch.hub.load('.', 'custom', path=model, source='local') 

        pass

    def _new(self, model: str):
        # load model from config file
        root_path = os.path.dirname(os.path.abspath(__file__))
        default_yaml_path = os.path.join(root_path, "cfg/yolov8.yaml")
        self.cfg = yaml_load(default_yaml_path)
        self.cfg["scale"] = guess_model_scale(model)
        self.model = DetectionModel(self.cfg)

        # Init weights, biases
        initialize_weights(self)