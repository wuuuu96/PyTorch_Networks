### 环境python==3.10 pytorch==2.3.1 cuda==12.1
### model运行结果: input:1x28x28    Total params:61,706
### model运行结果: input:3x32x32    Total params:83,126
### 论文中model训练结果: Sigmoid + Avgpool2d + 没有Dropout epoch=100 val_acc=87.34%
### 改进model训练结果_1: ReLU + MaxPool + Dropout + AdamW + OneCycleLR + label smoothing epoch=100 val_acc=89.68%
### 1.视频讲解

- 📺:[LeNet-5网络诞生背景](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=32)

- 📺:[LeNet-5网络参数详解](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=33)

<img width="1444" height="752" alt="image" src="https://github.com/user-attachments/assets/e57b9cc7-7bad-4102-b92f-f9f919f276ca" />

<img width="1296" height="633" alt="image" src="https://github.com/user-attachments/assets/e8638ca5-de24-4c0b-b7c0-876e4c1d1008" />

- 📺:[LeNet-5总结](https://www.bilibili.com/video/BV1e34y1M7wR?spm_id_from=333.788.videopod.episodes&vd_source=ee0d4853ce8dacb7fdfc07d40e328f36&p=34)

<img width="1319" height="659" alt="image" src="https://github.com/user-attachments/assets/ad648c98-c41e-4a5e-8abb-d5c0b56ef88b" />

### 2.LeNet-5模型搭建

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


    **# 如果训练3x32x32的图像，要改3个地方，
    # 第一卷积层 in_channels = 1  => in_channels = 3
    # self.fc1 = nn.Linear(400, 120) => self.fc1 = nn.Linear(576, 120) 这里576是每一层计算出的结果
    # print(summary(model,(1,28,28))) => print(summary(model,(3,32,32)))**

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

```

* ##### 2.4 主函数运行结果

<img width="798" height="573" alt="image" src="https://github.com/user-attachments/assets/c372404d-b10c-4f5e-9ccc-5b38506237f1" />

### 3.LeNet-5模型训练(导包+数据加载+训练和验证+可视化+主函数)

* ##### 3.1 导包

``` pyhon
import copy
import time
import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
from model import LeNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
```

* ##### 3.2 数据加载

``` python
def train_val_process():

    transform = transforms.Compose([
        transforms.RandomCrop(28,2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])


    full_data = FashionMNIST(root='./data',train=True,transform=transform,download=True)

    train_data,val_data = data.random_split(full_data,[round(0.8*len(full_data)),round(0.2*len(full_data))])

    train_dataloader = data.DataLoader(train_data,32,True,num_workers=0)
    val_dataloader = data.DataLoader(val_data,32,False,num_workers=0)

    return train_dataloader,val_dataloader
```

* ##### 3.3 训练和验证

``` python

def train_model(model,train_dataloader,val_dataloader,num_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_all = []

    train_acc_all = []

    val_loss_all = []

    val_acc_all =[]

    since = time.time()

    for epoch in range(num_epoch):
        print(f'Epoch {epoch+1}/{num_epoch}')
        print('-'*30)

        epoch_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0
        val_loss = 0.0
        val_acc = 0.0
        val_num = 0

        model.train()
        t1 = time.time()
        for step,(inputs,targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)



            outputs = model(inputs)

            pred = torch.argmax(outputs,dim=1)

            loss = criterion(outputs,targets)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*inputs.size(0)

            train_acc += torch.sum(pred==targets).item()

            train_num += inputs.size(0)

        t2 = time.time()
        model.eval()
        with torch.no_grad():
            for step,(inputs,targets) in enumerate(val_dataloader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                pred =torch.argmax(outputs,dim=1)

                loss = criterion(outputs,targets)

                val_loss += loss.item()*inputs.size(0)

                val_acc += torch.sum(pred==targets).item()

                val_num += inputs.size(0)
        t3 = time.time()

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc / val_num)

        print(f"Epoch:{epoch+1} Train_loss:{train_loss_all[-1]:.4f} Train_acc:{train_acc_all[-1]*100:.2f}")
        print(f"Epoch:{epoch+1} Val_loss:{val_loss_all[-1]:.4f} Val_acc:{val_acc_all[-1] * 100:.2f}")
        print(f"训练时间:{t2 - t1:.2f} 验证时间:{t3 - t2:.2f} 总用时:{time.time()-epoch_time:.2f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),'./model.pth')

            print(f"验证准确率为:{best_acc*100:.2f}")

    total = time.time() - since

    print(f"\n 训练完成，总用时为:{total//60:.0f}m{total%60:.0f}s")
    print(f"验证准确率为:{best_acc*100:.2f}")

    model.load_state_dict(best_model_wts)

    train_process = pd.DataFrame(data={"epoch":range(1,num_epoch+1),
                                       "train_loss_all":train_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_loss_all":val_loss_all,
                                       "val_acc_all":val_acc_all
                                       })
    return train_process

```

* ##### 3.4 可视化

``` python

def plot(train_process):
    plt.figure(figsize=(12,4))

    plt.subplot(121)

    plt.plot(train_process['epoch'],train_process['train_loss_all'],'-ro',label='train_loss')
    plt.plot(train_process['epoch'],train_process['val_loss_all'],'-ro',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(train_process['epoch'],train_process['train_acc_all'],'-ro',label='train_acc')
    plt.plot(train_process['epoch'],train_process['val_acc_all'],'-ro',label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.show()

```

* ##### 3.5 主函数

``` python
if __name__ == '__main__':
    model = LeNet()

    train_dataloader,val_dataloader = train_val_process()

    train_process = train_model(model,train_dataloader,val_dataloader,100)

    plot(train_process)

```
