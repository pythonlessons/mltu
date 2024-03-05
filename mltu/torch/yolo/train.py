import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
# from ultralytics.data.dataset import YOLODataset
from mltu.torch.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.torch.model import Model
# model = YOLO('yolov8n.pt')
from metrics import YoloMetrics
from mltu.transformers import ImageResizer, ImageShowCV2
from mltu.annotations.detections import Detections, Detection
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen, \
    RandomMirror, RandomFlip, RandomGaussianBlur, RandomSaltAndPepper, RandomDropBlock, RandomMosaic
import torch.nn as nn

from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, Model2onnx, WarmupCosineDecay
# from loss import v8DetectionLoss


train_path = "/home/rokbal/Personal/ultralytics/generated_data/train/images"
train_labels = "/home/rokbal/Personal/ultralytics/generated_data/train/labels"

val_path = "/home/rokbal/Personal/ultralytics/generated_data/valid/images"
val_labels = "/home/rokbal/Personal/ultralytics/generated_data/valid/labels"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
for f in tqdm(os.listdir(train_path)):
    img_path = Path(os.path.join(train_path, f))
    txt_path = Path(os.path.join(train_labels, img_path.stem + ".txt"))
    if txt_path.exists():
        dataset.append([str(img_path), str(txt_path)])

val_dataset = []
for f in tqdm(os.listdir(val_path)):
    img_path = Path(os.path.join(val_path, f))
    txt_path = Path(os.path.join(val_labels, img_path.stem + ".txt"))
    if txt_path.exists():
        val_dataset.append([str(img_path), str(txt_path)])

def annotation_preprocessor(image, annotation):
    with open(annotation, "r") as f:
        data = f.readlines()

    dets = [Detection(
            bbox=[float(x) for x in line.split()[1:]],
            label="ID",
            confidence=1.0,
            relative=True
        ) for line in data]

    detections = Detections(
        labels={0: "ID"},
        width=image.width,
        height=image.height,
        image_path=image.path,
        detections=dets
    )

    return image, detections


def image_normalizer(image, annotation):
    img = image.numpy().transpose(2, 0, 1) / 255.0
    return img, annotation


class YoloPreprocessor:
    def __init__(self, imgsz=640):
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

        batch["cls"] = torch.tensor(batch["cls"]).to(device)
        batch["bboxes"] = torch.tensor(batch["bboxes"]).to(device)
        batch["batch_idx"] = torch.tensor(batch["batch_idx"]).to(device)

        return np.array(images), batch





img_size = 320
# Create a data provider for the dataset
train_data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=16,
    data_preprocessors=[
        ImageReader(CVImage),
        annotation_preprocessor,
        ],
    transformers=[
        ImageShowCV2(),
        ImageResizer(img_size, img_size),
        image_normalizer,
    ],
    use_cache=True,
    batch_postprocessors=[
        YoloPreprocessor(img_size)
    ],
    use_multiprocessing=False,
    max_queue_size=1,
    workers=1,
    numpy=False,
    augmentors=[
        # RandomBrightness(), 
        # RandomErodeDilate(),
        # RandomSharpen(),
        # RandomMirror(0.5),
        # RandomFlip(0.5),
        # RandomGaussianBlur(),
        # RandomSaltAndPepper(),
        # RandomRotate(0.5, angle=10),
        # RandomDropBlock(0.5),
        RandomMosaic(random_chance=1.0)
    ]
)

for d in tqdm(train_data_provider):
    pass


# train_data_provider, val_data_provider = data_provider.split(0.8)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomMirror(0.5),
    RandomFlip(0.5),
    RandomGaussianBlur(),
    RandomSaltAndPepper(),

    # RandomRotate(angle=10), 
    # ImageShowCV2(),
]

val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=16,
    data_preprocessors=[
        ImageReader(CVImage),
        annotation_preprocessor,
        ],
    transformers=[
        # ImageShowCV2(),
        ImageResizer(img_size, img_size),
        image_normalizer,
    ],
    use_cache=True,
    batch_postprocessors=[
        YoloPreprocessor(img_size)
    ],
    use_multiprocessing=False,
    max_queue_size=32,
    # workers=32,
    numpy=False,
)


# while True:
#     start = time.time()
#     for batch in tqdm(train_data_provider):
#         pass
#     end = time.time()
#     fps = len(data_provider) / (end - start)
#     print(f"FPS: {fps:.2f}")

# for i in range(1000):
#     pbar = tqdm(train_data_provider)
#     for d in pbar:
#         pass


# # model_y = YOLO('yolov8n.yaml')
# model = DetectionModel('yolov8n.yaml', nc=1) # YOLO('yolov8n.pt')
# model.to(device)
# model.load_state_dict(torch.load("Models/detector/1707665153/model.pt"))

# for k, v in model.named_parameters():
#     if any(x in k for x in [".dfl"]):
#         print("freezing", k)
#         v.requires_grad = False
#     elif not v.requires_grad:
#         v.requires_grad = True

# # model.model.nc = 1
# # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.937, 0.999))

# pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
# for k, v in model.named_modules():
#     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
#         pg2.append(v.bias)  # biases
#     if isinstance(v, nn.BatchNorm2d):
#         pg0.append(v.weight)  # no decay
#     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
#         pg1.append(v.weight)  # apply decay

# optimizer = torch.optim.AdamW(pg0, lr=1e-2, betas=(0.937, 0.999))
# optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})  # add pg1 with weight_decay
# optimizer.add_param_group({'params': pg2, 'weight_decay': 0.0})  # add pg2 (biases)
# # del pg0, pg1, pg2



# loss = v8DetectionLoss(model)
# # create model object that will handle training and testing of the network
# model = Model(
#     model, 
#     optimizer, 
#     loss, 
#     metrics=[YoloMetrics(nc=1)],
#     log_errors=False,
#     mixed_precision=False,
#     output_path=f"Models/detector/{int(time.time())}",
#     clip_grad_norm=10.0,
#     ema=True,
# )

# # ouput_path = f"Models/detector/{int(time.time())}"
# # save_model_path = ouput_path + "/model.pt"
# # modelCheckpoint = ModelCheckpoint(save_model_path, monitor="val_fitness", mode="max", save_best_only=True, verbose=True)
# # tensorBoard = TensorBoard(log_dir=ouput_path)
# # earlyStopping = EarlyStopping(monitor="val_fitness", mode="max", patience=21, verbose=True)
# # reduceLROnPlateau = ReduceLROnPlateau(monitor="val_fitness", factor=0.9, mode="max", patience=5, verbose=True)
# # model2onnx = Model2onnx(modelCheckpoint.filepath, input_shape=(1, 3, img_size, img_size), verbose=True, opset_version=14, dynamic_axes=None)
# # warmupCosineDecay = WarmupCosineDecay(lr_after_warmup=1e-3, final_lr=1e-3, warmup_epochs=5, decay_epochs=0, initial_lr=1e-5)

# # history = model.fit(
# #     train_data_provider,
# #     test_dataProvider=val_data_provider,
# #     epochs=1, 
# #     callbacks=[modelCheckpoint, tensorBoard, earlyStopping, model2onnx, warmupCosineDecay]
# #     )
# # for _ in data_provider:
# #     pass


# # data = {
# #     "train": train_path,
# # }

# # dataset = YOLODataset(
# #     img_path=train_path,
# #     imgsz=320,
# #     batch_size=4,
# # )


# import math
# from mltu.torch.yolo.tolov8_utils import replace_c2f_with_c2f_v2
# import torch_pruning as tp
# from ultralytics.nn.modules import Detect


# example_inputs = torch.randn(1, 3, img_size, img_size).to(model.device)
# macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
# replace_c2f_with_c2f_v2(model.model)
# model.model = model.model.to(model.device)
# model.model.train()
# base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)
# macs_list.append(base_macs)

# # prune same ratio of filter based on initial size
# target_prune_rate = 0.5
# iterative_steps = 16
# pruning_ratio = 1 - math.pow((1 - target_prune_rate), 1 / iterative_steps)

# init_map = 0
# print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

# ignored_layers = [m for m in model.model.modules() if isinstance(m, (Detect,))]
# output_path = model.output_path
# for i in range(iterative_steps):

#     model.model.train()
#     for name, param in model.model.named_parameters():
#         param.requires_grad = True


#     example_inputs = example_inputs.to(model.device)
#     pruner = tp.pruner.GroupNormPruner(
#         model.model,
#         example_inputs,
#         importance=tp.importance.GroupNormImportance(),  # L2 norm pruning,
#         iterative_steps=1,
#         pruning_ratio=pruning_ratio,
#         ignored_layers=ignored_layers,
#         unwrapped_parameters=[]
#     )

#     pruner.step()

#     pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
#     macs_list.append(pruned_macs)
#     current_speed_up = float(macs_list[0]) / pruned_macs
#     print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, speed up={current_speed_up}")

#     # os.makedirs(ouput_path, exist_ok=True)
#     #save_model_path = ouput_path + f"/model_prune{i}.pth"
#     model.output_path = Path(os.path.join(output_path.parent, f"prune_{i}", output_path.name))
#     history = model.fit(
#         train_data_provider,
#         test_dataProvider=val_data_provider,
#         epochs=10, 
#         callbacks=[
#             TensorBoard(),
#             ModelCheckpoint(monitor="val_fitness", mode="max", save_best_only=True, verbose=True),
#             Model2onnx(input_shape=(1, 3, img_size, img_size), verbose=True, opset_version=14, dynamic_axes=None),
#             # WarmupCosineDecay(lr_after_warmup=1e-3, final_lr=1e-4, warmup_epochs=3, decay_epochs=7, initial_lr=1e-5)
#             #modelCheckpoint, tensorBoard, earlyStopping, model2onnx, warmupCosineDecay
#             ],
#         )
    
#     # reset model optimizer
#     # model.optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3, betas=(0.937, 0.999))
#     # model.optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})  # add pg1 with weight_decay
#     # model.optimizer.add_param_group({'params': pg2, 'weight_decay': 0.0})  # add pg2 (biases)

#     # torch.save(model.model, save_model_path)
#     model.model.to(model.device)
#     del pruner