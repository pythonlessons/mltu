import os
import stow
import numpy as np
import onnxruntime as ort

class OnnxInferenceModel:
    """ Base class for all inference models that use onnxruntime 

    Attributes:
        model_path (str, optional): Path to the model folder. Defaults to "".
        force_cpu (bool, optional): Force the model to run on CPU or GPU. Defaults to GPU.
        default_model_name (str, optional): Default model name. Defaults to "model.onnx".
    """
    def __init__(
        self, 
        model_path: str = "",
        force_cpu: bool = False,
        default_model_name: str = "model.onnx"
        ):
        
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.default_model_name = default_model_name

        if isinstance(stow.artefact(self.model_path), stow.Directory):
            self.model_path = stow.join(self.model_path, self.default_model_name)

        if not os.path.exists(self.model_path):
            raise Exception(f"Model path ({self.model_path}) does not exist")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == "GPU" and not force_cpu else ['CPUExecutionProvider']

        self.model = ort.InferenceSession(self.model_path, providers=providers)

        # Update providers priority to only CPUExecutionProvider
        if self.force_cpu:
            self.model.set_providers(['CPUExecutionProvider'])

        self.input_shape = self.model.get_inputs()[0].shape[1:]
        self.input_name = self.model._inputs_meta[0].name
        self.output_name = self.model._outputs_meta[0].name

    def predict(self, image: np.ndarray):
        raise NotImplementedError

    def __call__(self, image: np.ndarray):
        return self.predict(image)