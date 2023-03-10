import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import torch
import torch.optim as optim

from mltu.torch.dataProvider import DataProvider
from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.metrics import CERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate

from model import CaptchaModel
from configs import ModelConfigs

def download_and_unzip(url, extract_to='Datasets'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

if not os.path.exists(os.path.join('Datasets', 'captcha_images_v2')):
    download_and_unzip('https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip', extract_to='Datasets')

# Create a list of all the images and labels in the dataset
dataset, vocab, max_len = [], set(), 0
captcha_path = os.path.join('Datasets', 'captcha_images_v2')
for file in os.listdir(captcha_path):
    file_path = os.path.join(captcha_path, file)
    # Get the file name without the extension
    file_text = os.path.splitext(file)[0]
    dataset.append([file_path, file_text])
    vocab.update(list(file_text))
    max_len = max(max_len, len(file_text))

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader()],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
)
# Split the dataset into training and validation sets
train_dataProvider, test_dataProvider = data_provider.split(split = 0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

network = CaptchaModel(len(configs.vocab))
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=0.001)

# put on cuda device if available
if torch.cuda.is_available():
    network = network.cuda()

# create callbacks
earlyStopping = EarlyStopping(monitor='val_CER', patience=50, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint('Models/08_/model.pt', monitor='val_CER', mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard('Models/08_/logs')
model2onnx = Model2onnx(
    saved_model_path='Models/08_/model.pt', 
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab)])
model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=1, 
    callbacks=[earlyStopping, tb_callback, model2onnx]
    )

# # Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, 'train.csv'))
test_dataProvider.to_csv(os.path.join(configs.model_path, 'val.csv'))