### 环境python==3.10 pytorch==2.3.1 cuda==12.1
### 主函数运行结果 input:1x28x28 Total params:61,706
### 主函数运行结果 input:3x32x32 Total params:83,126
#### 1.视频讲解

- 📺:[LeNet-5网络诞生背景](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=32)

- 📺:[LeNet-5网络参数详解](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=33)

<img width="1444" height="752" alt="image" src="https://github.com/user-attachments/assets/e57b9cc7-7bad-4102-b92f-f9f919f276ca" />

<img width="1296" height="633" alt="image" src="https://github.com/user-attachments/assets/e8638ca5-de24-4c0b-b7c0-876e4c1d1008" />

- 📺:[LeNet-5总结](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=34)

<img width="1319" height="659" alt="image" src="https://github.com/user-attachments/assets/ad648c98-c41e-4a5e-8abb-d5c0b56ef88b" />

#### 2.LeNet-5模型搭建

**依据前面视频讲解的LeNet-5网络参数来搭建网络**

<img width="1296" height="633" alt="image" src="https://github.com/user-attachments/assets/e8638ca5-de24-4c0b-b7c0-876e4c1d1008" />

* ##### 2.1 导包

``` python
import torch
from torch import nn
from torchsummary import summary
```

* ##### 2.2 定义Lenet模块类
![微信图片_20250902153651_46_28](https://github.com/user-attachments/assets/0efec7d0-22ac-49d6-aba9-f12d4f391e11)
![微信图片_20250902153656_47_28](https://github.com/user-attachments/assets/946bf3e5-c89e-4c48-a1ce-e3818f481352)

``` pyhon
#初始化
class Lenet(nn.Module):
  def __init__(self):
    super().__init__()
#定义层
    #第一卷积层 in_channels = 1 , out_channels = 6 , kernel_size = 5 , stride = 1 , padding = 2
    #in_channels = 1是因为输入的图像为单通道，即input:28*28*1，如果是彩色图，即28*28*3 应该改in_channels为3
    #out_channels = 6是因为卷积核个数为6，kernel_size和stride和padding设置要不改变图像W和H


    # 如果训练3x32x32的图像，要改3个地方，
    # 第一卷积层 in_channels = 1  => in_channels = 3
    # self.fc1 = nn.Linear(400, 120) => self.fc1 = nn.Linear(576, 120) 这里576是每一层计算出的结果
    # print(summary(model,(1,28,28))) => print(summary(model,(3,32,32)))

    self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
    self.sig = nn.Sigmoid()  # Sigmoid激活函数
    self.pool = nn.AvgPool2d(2, 2)  # 平均池化
    self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)  # stride=1和padding=0可以省略


    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(400, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
#前向传播
    def forward(self, x):
        x = self.pool(self.sig(self.conv1(x)))
        x = self.pool(self.sig(self.conv2(x)))
        x = self.flatten(x)
        x = self.sig(self.fc1(x))
        x = self.sig(self.fc2(x))
        x = self.fc3(x)
        return x
* ##### 2.3 主函数
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))

* ##### 2.4 主函数运行结果
```
<img width="798" height="573" alt="image" src="https://github.com/user-attachments/assets/c372404d-b10c-4f5e-9ccc-5b38506237f1" />

