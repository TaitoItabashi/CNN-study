"""キャッシュを使って手書き画像を生成するプログラム"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from PixelCNN import MaskedConv2d ,PixelCNN 


num_of_layers = 7 # 畳み込み層の数
num_of_channels = 64 # チャネル数
num_of_epochs = 50
X_DIM = 28
NUM_OF_VALUES = 255
device = 'cuda:0'


pixel_cnn = PixelCNN(num_of_channels, num_of_layers).to(device)
pixel_cnn.load_state_dict(torch.load('./cache/pixelcnn_weight.pth'))


sample = torch.Tensor(25, 1, X_DIM, X_DIM).to(device)
pixel_cnn.eval()
sample.fill_(0)
for i in range(X_DIM):
    for j in range(X_DIM):
        out = pixel_cnn(sample).to(device)
        probs = F.softmax(out[:, :, i, j], dim=1)
        sample[:, :, i, j] = torch.multinomial(probs, 1).float() / NUM_OF_VALUES
 
sample_array = sample.cpu().numpy().squeeze()
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(5, 5))
for i in range(25):
  idx = divmod(i, 5)
  ax[idx].imshow(sample_array[i]*255, cmap='gray')
  ax[idx].axis('off');
fig.savefig('./image/generate.png')
fig.show()