import os
import cv2
import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model import Net

# define path to store dataset
path = "Datasets/mnist"

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

# uncomment to show images from dataset using OpenCV
# for train_image, train_target in zip(train_data, train_targets):
#     train_image = cv2.resize(train_image, (400, 400))
#     cv2.imshow("Image", train_image)
#     # if Q button break this loop
#     if cv2.waitKey(0) & 0xFF == ord("q"):
#         break
# cv2.destroyAllWindows()

# define training hyperparameters
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

# create network
network = Net()

# uncomment to print network summary
summary(network, (1, 28, 28), device="cpu")

# define loss function and optimizer
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# create training loop
def train(epoch):
    # set network to training mode
    network.train()

    loss_sum = 0
    # create a progress bar
    train_pbar = tqdm(zip(train_batches, train_target_batches), total=len(train_batches))
    for index, (data, target) in enumerate(train_pbar, start=1):

        # convert data to torch.FloatTensor
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = network(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        # update progress bar with loss value
        loss_sum += loss.item()
        train_pbar.set_description(f"Epoch {epoch}, loss: {loss_sum / index:.4f}")

# create testing loop
def test(epoch):
    # set network to evaluation mode
    network.eval()

    correct, loss_sum = 0, 0
    # create progress bar
    val_pbar = tqdm(zip(test_batches, test_target_batches), total=len(test_batches))
    with torch.no_grad():
        for index, (data, target) in enumerate(val_pbar, start=1):

            # convert data to torch.FloatTensor
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).long()

            # forward pass
            output = network(data)

            # update progress bar with loss and accuracy values
            loss_sum += loss_function(output, target).item() / target.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum() / target.size(0)

            val_pbar.set_description(f"val_loss: {loss_sum / index:.4f}, val_accuracy: {correct / index:.4f}")


# train and test the model
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

# define output path and create folder if not exists
output_path = "Models/06_pytorch_introduction"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save model.pt to defined output path
torch.save(network.state_dict(), os.path.join(output_path, "model.pt"))