# Run Ultralytics YOLOv8 pretrained model

YouTube tutorial link:
- [YOLOv8: Real-Time Object Detection Simplified](https://youtu.be/vegL__weCxY);
- [YOLOv8: Customizing Object Detector training](https://youtu.be/ysYiV1CbCyY);

First, I recommend you to install the required packages in a virtual environment:
```bash
mltu==1.2.5
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

## Customize YoloV8 Object Detector training:
```python
import os
import time
import torch
from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, ImageShowCV2, ImageNormalizer
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen, \
    RandomMirror, RandomFlip, RandomGaussianBlur, RandomSaltAndPepper, RandomDropBlock, RandomMosaic, RandomElasticTransform
from mltu.torch.model import Model
from mltu.torch.dataProvider import DataProvider
from mltu.torch.yolo.annotation import VOCAnnotationReader
from mltu.torch.yolo.preprocessors import YoloPreprocessor
from mltu.torch.yolo.loss import v8DetectionLoss
from mltu.torch.yolo.metrics import YoloMetrics
from mltu.torch.yolo.optimizer import build_optimizer, AccumulativeOptimizer
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay

from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.model import Model as BaseModel

# https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
annotations_path = "Datasets/car-plate-detection/annotations"

# Create a dataset from the annotations, the dataset is a list of lists where each list contains the [image path, annotation path]
dataset = [[None, os.path.join(annotations_path, f)] for f in os.listdir(annotations_path)]

# Make sure torch can see GPU device, it is not recommended to train with CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_size = 416
labels = {0: "licence"}

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=16,
    data_preprocessors=[
        VOCAnnotationReader(labels=labels),
        ImageReader(CVImage),
        ],
    transformers=[
        # ImageShowCV2(),
        ImageResizer(img_size, img_size),
        ImageNormalizer(transpose_axis=True),
    ],
    batch_postprocessors=[
        YoloPreprocessor(device, img_size)
    ],
    numpy=False,
)

# split the dataset into train and test
train_data_provider, val_data_provider = data_provider.split(0.9, shuffle=False)

# Attaach augmentation to the train data provider
train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomMirror(),
    RandomFlip(),
    RandomElasticTransform(),
    RandomGaussianBlur(),
    RandomSaltAndPepper(),
    RandomRotate(angle=10),
    RandomDropBlock(),
    RandomMosaic(),
]

base_model = BaseModel("yolov8n.pt")
# Create a YOLO model
model = DetectionModel('yolov8n.yaml', nc=len(labels))

# Load the weight from base model
try: model.load_state_dict(base_model.model.state_dict(), strict=False)
except: pass

model.to(device)

for k, v in model.named_parameters():
    if any(x in k for x in [".dfl"]):
        print("freezing", k)
        v.requires_grad = False
    elif not v.requires_grad:
        v.requires_grad = True

lr = 1e-3
optimizer = build_optimizer(model.model, name="AdamW", lr=lr, weight_decay=0.0, momentum=0.937, decay=0.0005)
optimizer = AccumulativeOptimizer(optimizer, 16, 64)

# create model object that will handle training and testing of the network
model = Model(
    model, 
    optimizer, 
    v8DetectionLoss(model), 
    metrics=[YoloMetrics(nc=len(labels))],
    log_errors=False,
    output_path=f"Models/11_Yolov8/{int(time.time())}",
    clip_grad_norm=10.0,
    ema=True,
)

modelCheckpoint = ModelCheckpoint(monitor="val_fitness", mode="max", save_best_only=True, verbose=True)
tensorBoard = TensorBoard()
earlyStopping = EarlyStopping(monitor="val_fitness", mode="max", patience=31, verbose=True)
model2onnx = Model2onnx(input_shape=(1, 3, img_size, img_size), verbose=True, opset_version=14, 
                    dynamic_axes = {"input": {0: "batch_size", 2: "height", 3: "width"}, 
                                    "output": {0: "batch_size", 2: "anchors"}},
                    metadata={"classes": labels})
warmupCosineDecayBias = WarmupCosineDecay(lr_after_warmup=lr, final_lr=lr, initial_lr=0.1, 
                                      warmup_steps=len(train_data_provider), warmup_epochs=10, ignore_param_groups=[1, 2]) # lr0
warmupCosineDecay = WarmupCosineDecay(lr_after_warmup=lr, final_lr=lr/10, initial_lr=1e-7, 
                                      warmup_steps=len(train_data_provider), warmup_epochs=10, decay_epochs=190, ignore_param_groups=[0]) # lr1 and lr2

# Train the model
history = model.fit(
    train_data_provider,
    test_dataProvider=val_data_provider,
    epochs=200, 
    callbacks=[
        modelCheckpoint, 
        tensorBoard, 
        earlyStopping, 
        model2onnx,
        warmupCosineDecayBias,
        warmupCosineDecay
        ]
    )
```

## Test Custom trained YoloV8 Object Detector:
```python
import os
import cv2
from mltu.annotations.detections import Detections
from mltu.torch.yolo.detectors.onnx_detector import Detector as OnnxDetector

# https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
images_path = "Datasets/car-plate-detection/images"

input_width, input_height = 416, 416
confidence_threshold = 0.5
iou_threshold = 0.5

detector = OnnxDetector("Models/11_Yolov8/1714135287/model.onnx", input_width, input_height, confidence_threshold, iou_threshold, force_cpu=False)

for image_path in os.listdir(images_path):

    frame = cv2.imread(os.path.join(images_path, image_path))

    # Perform Yolo object detection
    detections: Detections = detector(frame)

    # Apply the detections to the frame
    frame = detections.applyToFrame(frame)

    # Print the FPS
    print(detector.fps)

    # Display the output image
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```