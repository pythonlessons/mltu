import os
import numpy as np
import requests, gzip, os, hashlib

import torch
import torch.optim as optim

from model import Net

from mltu.torch.dataProvider import DataProvider
from mltu.torch.model import Model
from mltu.torch.metrics import Accuracy
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint

# define path to store dataset
path = "Datasets/data"

def fetch(url):
    if os.path.exists(path) is False:
        os.makedirs(path)

    fp = os.path.join(path, hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

# load mnist dataset from yann.lecun.com, train data is of shape (60000, 28, 28) and targets are of shape (60000)
train_data = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
train_targets = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
test_data = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

train_dataset = [[data, target] for data, target in zip(train_data, train_targets)]
test_dataset = [[data, target] for data, target in zip(test_data, test_targets)]

def preprocessor(data, target):
    # original data is shape of (28, 28), expand to (1, 28, 28) and normalize to [0, 1]
    data = np.expand_dims(data, axis=0) / 255.0
    return data, target

train_dataProvider = DataProvider(
    train_dataset, 
    data_preprocessors=[preprocessor],
    batch_size=64,
    )

test_dataProvider = DataProvider(
    test_dataset,
    data_preprocessors=[preprocessor],
    batch_size=64
    )

# create network, optimizer and define loss function
network = Net()
optimizer = optim.Adam(network.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

# put on cuda device if available
if torch.cuda.is_available():
    network = network.cuda()

# create callbacks
earlyStopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3, 
    mode="max", 
    verbose=1
    )
modelCheckpoint = ModelCheckpoint(
    "Models/07_pytorch_wrapper/model.pt",
    monitor="val_accuracy",
    mode="max", 
    save_best_only=True, 
    verbose=1
    )

# create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[Accuracy()])
model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=100, 
    callbacks=[earlyStopping, modelCheckpoint]
    )