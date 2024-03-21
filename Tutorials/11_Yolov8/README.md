# Run Ultralytics YOLOv8 pretrained model

YouTube tutorial link: [YOLOv8: Real-Time Object Detection Simplified](https://youtu.be/vegL__weCxY)

First, I recommend you to install the required packages in a virtual environment:
```bash
mltu==1.2.3
ultralytics==8.1.28
torch==2.0.0
torchvision==0.15.1
onnxruntime==1.15.1
onnx==1.12.0
```

### Run the pretrained Ultralytics YOLOv8 within torch environment on webcam:
```python
import cv2
from ultralytics.engine.model import Model as BaseModel
from mltu.torch.yolo.detectors.torch_detector import Detector as TorchDetector

input_width, input_height = 640, 640
confidence_threshold = 0.5
iou_threshold = 0.5

base_model = BaseModel("yolov8m.pt")
detector = TorchDetector(base_model.model, input_width, input_height, base_model.names, confidence_threshold, iou_threshold)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Yolo object detection
    detections = detector(frame)

    # Apply the detections to the frame
    frame = detections.applyToFrame(frame)

    # Print the FPS
    print(detector.fps)

    # Display the output image
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Convert the pretrained model to ONNX:
```python
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
```

## Run the YOLOv8 ONNX model with ONNX Runtime:
```python
import cv2
from mltu.torch.yolo.detectors.onnx_detector import Detector as OnnxDetector

input_width, input_height = 640, 640
confidence_threshold = 0.5
iou_threshold = 0.5

detector = OnnxDetector("yolov8m.onnx", input_width, input_height, confidence_threshold, iou_threshold)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Yolo object detection
    detections = detector(frame)

    # Apply the detections to the frame
    frame = detections.applyToFrame(frame)

    # Print the FPS
    print(detector.fps)

    # Display the output image
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```