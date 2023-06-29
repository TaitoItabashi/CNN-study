"""モデルを使って学習、キャッシュを作成するプログラム"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchinfo import summary
from PixelCNN import MaskedConv2d, PixelCNN
    
    
trainloader = data.DataLoader(datasets.MNIST('data', train=True,
                                             download=True,
                                             transform=transforms.ToTensor()),
                              batch_size=128, shuffle=True,
                              num_workers=1, pin_memory=True)
 
testloader = data.DataLoader(datasets.MNIST('data', train=False,
                                            download=True,
                                            transform=transforms.ToTensor()),
                             batch_size=128, shuffle=False,
                             num_workers=1, pin_memory=True)


num_of_layers = 7 # 畳み込み層の数
num_of_channels = 64 # チャネル数
num_of_epochs = 5
X_DIM = 28
NUM_OF_VALUES = 255
device = 'cuda:0'


pixel_cnn = PixelCNN(num_of_channels, num_of_layers).to(device)
summary(model=pixel_cnn, input_size=(128, 1, 28, 28))


optimizer = optim.Adam(list(pixel_cnn.parameters()))
criterion = nn.CrossEntropyLoss()
train_losses, test_losses = [], []
for epoch in range(num_of_epochs):
    # 学習
    train_errors = []
    pixel_cnn.train()
    for x, label in trainloader:
        x = x.to(device)
        target = (x[:,0] * NUM_OF_VALUES).long()
        loss = criterion(pixel_cnn(x), target)
        train_errors.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
    # 評価
    with torch.no_grad():
        test_errors = []
        pixel_cnn.eval()
        for x, label in testloader:
            x = x.to(device)
            target = (x[:,0] * NUM_OF_VALUES).long()
            loss = criterion(pixel_cnn(x), target)
            test_errors.append(loss.item())
         
        print(f'epoch: {epoch}/{num_of_epochs} train error: {np.mean(train_errors):0.3f} \
              test error {np.mean(test_errors):0.3f}')
    train_losses.append(np.mean(train_errors))
    test_losses.append(np.mean(test_errors))

torch.save(pixel_cnn.state_dict(), './cache/pixelcnn_weight.pth')
torch.save(pixel_cnn, './cache/pixelcnn_model.pth')