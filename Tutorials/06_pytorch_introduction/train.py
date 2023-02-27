import os
import cv2
import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Net

path='Datasets/data'
def fetch(url):
    if os.path.exists(path) is False:
        os.makedirs(path)

    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
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

# uncomment to show images from dataset using OpenCV
# for train_image, train_target in zip(train_data, train_targets):
#     train_image = cv2.resize(train_image, (300, 300))
#     cv2.imshow("Image", train_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# define hyperparameters
n_epochs = 5
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.001

# reshape data to (items, channels, height, width) and normalize to [0, 1]
train_data = np.expand_dims(train_data, axis=1) / 255.0
test_data = np.expand_dims(test_data, axis=1) / 255.0

# split data into batches of size [(batch_size, 1, 28, 28) ...]
train_batches = [np.array(train_data[i:i+batch_size_train]) for i in range(0, len(train_data), batch_size_train)]
# split targets into batches of size [(batch_size) ...]
train_target_batches = [np.array(train_targets[i:i+batch_size_train]) for i in range(0, len(train_targets), batch_size_train)]

test_batches = [np.array(test_data[i:i+batch_size_test]) for i in range(0, len(test_data), batch_size_test)]
test_target_batches = [np.array(test_targets[i:i+batch_size_test]) for i in range(0, len(test_targets), batch_size_test)]

# create network and optimizer
network = Net()
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# create training loop
def train(epoch):
    network.train()

    loss_sum = 0
    train_pbar = tqdm(zip(train_batches, train_target_batches), total=len(train_batches))
    for data, target in train_pbar:

        # convert data to torch.FloatTensor
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        train_pbar.set_description(f"Epoch {epoch}, loss: {loss_sum / len(train_batches):.4f}")

# create testing loop
def test(epoch):
    network.eval()

    correct = 0
    loss_sum = 0
    val_pbar = tqdm(zip(test_batches, test_target_batches), total=len(test_batches))
    with torch.no_grad():
        for data, target in val_pbar:
            # convert data to torch.FloatTensor
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).long()

            output = network(data)
            loss_sum += F.nll_loss(output, target).item() / target.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum() / target.size(0)

            val_pbar.set_description(f"val_loss: {loss_sum / len(test_batches):.4f}, val_accuracy: {correct / len(test_batches):.4f}")

# train and test the model
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

# define output path and create folder if not exists
output_path = 'Models/06_pytorch_introduction'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save model.pt to defined output path
torch.save(network.state_dict(), os.path.join(output_path, "model.pt"))