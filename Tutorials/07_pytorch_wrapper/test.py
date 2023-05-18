import os
import cv2
import torch
import numpy as np
import requests, gzip, os, hashlib

from model import Net

path = "Datasets/mnist" # Path where to save the downloaded mnist dataset

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

test_data = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
test_targets = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

# output path
model_path = "Model/07_pytorch_wrapper"

# construct network and load weights
network = Net()
network.load_state_dict(torch.load("Models/07_pytorch_wrapper/model.pt"))
network.eval() # set to evaluation mode

# loop over test images
for test_image, test_target in zip(test_data, test_targets):

    # normalize image and convert to tensor
    inference_image = torch.from_numpy(test_image).float() / 255.0
    inference_image = inference_image.unsqueeze(0).unsqueeze(0)

    # predict
    output = network(inference_image)
    pred = output.argmax(dim=1, keepdim=True)
    prediction = str(pred.item())

    test_image = cv2.resize(test_image, (400, 400))
    cv2.imshow(prediction, test_image)
    key = cv2.waitKey(0)
    if key == ord("q"): # break on q key
        break
 
    cv2.destroyAllWindows()