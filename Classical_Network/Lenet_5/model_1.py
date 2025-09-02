import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # FashionMNIST 是 1 通道
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28x28 -> 28x28
        self.act   = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(2, 2)            # -> 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)           # -> 10x10
        # pool -> 5x5

        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16*5*5, 120)          # 400 -> 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.flatten(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        x = self.fc3(x)
        return x
