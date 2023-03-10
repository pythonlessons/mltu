import torch.nn as nn
import torch.nn.functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(768, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()

        # normalize images between 0 and 1
        images_flaot = images / 255.0

        # transpose image to channel first
        images_flaot = images_flaot.permute(0, 3, 1, 2)

        x = F.relu(self.conv_1(images_flaot))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        # x = x.view(-1, x.size(1), x.size(2) * x.size(3))
        # x = x.view(-1, x.size(1), -1)
        x = F.relu(self.linear_1(x))
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        x = self.output(x)
        # x = self.softmax(x)
        x = F.log_softmax(x, 2)

        return x