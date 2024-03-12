import os
import time
import torch
from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, ImageShowCV2, ImageNormalizer
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen, \
    RandomMirror, RandomFlip, RandomGaussianBlur, RandomSaltAndPepper, RandomDropBlock, RandomMosaic
from mltu.torch.model import Model
from mltu.torch.dataProvider import DataProvider
from mltu.torch.yolo.annotation import VOCAnnotationReader
from mltu.torch.yolo.preprocessors import YoloPreprocessor
from mltu.torch.yolo.loss import v8DetectionLoss
from mltu.torch.yolo.metrics import YoloMetrics
from mltu.torch.yolo.optimizer import build_optimizer, AccumulativeOptimizer
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Model2onnx, WarmupCosineDecay

from ultralytics.nn.tasks import DetectionModel
from ultralytics.engine.model import Model as BaseModel


annotations_path = "Datasets/car-plate-detection/annotations"

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

# for b in data_provider:
#     pass

# split the dataset into train and test
train_data_provider, val_data_provider = data_provider.split(0.9, shuffle=False)

# Attaach augmentation to the train data provider
train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomMirror(),
    RandomFlip(),
    RandomGaussianBlur(),
    RandomSaltAndPepper(),
    RandomRotate(angle=10),
    RandomDropBlock(),
    RandomMosaic(),
]

# for batch in train_data_provider:
#     pass
    # print(batch)
    # break



base_model = BaseModel("yolov8n.pt")
# Create a YOLO model
model = DetectionModel('yolov8n.yaml', nc=len(labels))

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
    output_path=f"Models/detector/{int(time.time())}",
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