import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from PIL import Image
from tqdm import tqdm

batch_size = 64
epochs = 10

class Net(nn.Module): #nn.Moduleを継承したクラスとして定義
    def __init__(self):
      super(Net,self).__init__()
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2) #畳み込み層:(入力チャンネル数, 出力チャンネル数、カーネルサイズ、パディング)
      self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

      self.fc1 = nn.Linear(32 * 7 * 7, 128) #全結合層
      self.fc2 = nn.Linear(128,10)

    def forward(self,x):
      x = torch.relu(F.max_pool2d(self.conv1(x), 2)) #畳み込み層の後のF.maxpool2dはプーリング層:(畳み込み層、ウィンドウサイズ)
      x = torch.relu(F.max_pool2d(self.conv2(x), 2))

      x = x.view(-1, x.size(1) * x.size(2) * x.size(3)) #テンソルを平らに(1次元に)する処理

      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x  

net = Net()
summary(model=net, input_size=(batch_size, 1, 28, 28))

#データローダーを作成
transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, ), (0.5, ))
                                                        ])

#MNISTデータ読み込み
train_dataset = dataset.MNIST(root='./data', download=True, train=True, transform=transform)

#データが多すぎるため少し減らす
train_size = int(len(train_dataset) * 0.1)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#損失関数、オプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

y_axis_list = []#損失プロット用y軸方向リスト

#訓練
for epoch in tqdm(range(epochs)):
  for batch, label in train_loader: 

        optimizer.zero_grad()

        outputs = net(batch)
        image = batch.numpy()
        #print(((image[0] + 1) * 255 / 2).astype(np.uint8).reshape(28, 28))
        """画像取得"""
        Image.fromarray(((image[0] + 1) * 255 / 2).astype(np.uint8).reshape(28, 28)).save("./image/image.png")
        loss = criterion(outputs,label)

        loss.backward()
        optimizer.step()

  y_axis_list.append(loss.detach().numpy())#プロット用のy軸方向リストに損失の値を代入

  print("epoch: %d  loss: %f" % (epoch+1 ,float(loss)))

x_axis_list = [num for num in range(10)]#損失プロット用x軸方向リスト

#損失の描画
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis_list,y_axis_list)
plt.savefig("result.png")
plt.show()