import torch
from ultralytics.engine.model import Model as BaseModel

base_model = BaseModel("yolov8m.pt")

classes = base_model.names
input_width, input_height = 640, 640
input_shape = (1, 3, input_width, input_height)
model = base_model.model

# place model on cpu
model.to("cpu")

# set the model to inference mode
model.eval()

# convert the model to ONNX format
dummy_input = torch.randn(input_shape).to("cpu")

# Export the model
torch.onnx.export(
    model,               
    dummy_input,                         
    "yolov8m.onnx",   
    export_params=True,        
    input_names = ["input"],   
    output_names = ["output"], 
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"}, 
        "output": {0: "batch_size", 2: "anchors"}
        }
)

# Add the class names to the model as metadata
import onnx

metadata = {"classes": classes}

# Load the ONNX model
onnx_model = onnx.load("yolov8m.onnx")

# Add the metadata dictionary to the onnx model's metadata_props attribute
for key, value in metadata.items():
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = str(value)

# Save the modified ONNX model
onnx.save(onnx_model, "yolov8m.onnx")