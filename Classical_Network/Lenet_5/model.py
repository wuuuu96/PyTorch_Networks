import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
      super().__init__()

      self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
      self.sig = nn.Sigmoid()  # Sigmoid激活函数
      self.pool = nn.AvgPool2d(2, 2)  # 平均池化
      self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)  # stride=1和padding=0可以省略

      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(400, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.sig(self.conv1(x)))
        x = self.pool(self.sig(self.conv2(x)))
        x = self.flatten(x)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))
