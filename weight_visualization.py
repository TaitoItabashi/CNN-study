# https://qiita.com/mathlive/items/d9f31f8538e20a102e14
# https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0/pytorch%E3%81%A7%E9%87%8D%E3%81%BF%E3%81%AE%E7%A2%BA%E8%AA%8D%E3%81%A8%E3%80%81%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF%E5%B1%A4%E3%81%AE%E3%82%AB%E3%83%BC%E3%83%8D%E3%83%AB%E3%81%AE%E5%8F%AF%E8%A6%96/

"""キャッシュに保存した重みを可視化するプログラム"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchinfo import summary
from PixelCNN import MaskedConv2d, PixelCNN


num_of_layers = 7 # 畳み込み層の数
num_of_channels = 64 # チャネル数
num_of_epochs = 50
X_DIM = 28
NUM_OF_VALUES = 255
device = 'cuda:0'


pixel_cnn = PixelCNN(num_of_channels, num_of_layers).to(device)
pixel_cnn.load_state_dict(torch.load('./cache/pixelcnn_weight.pth', map_location=device))
print(pixel_cnn.state_dict().keys()) 
print(pixel_cnn.state_dict()['layers.0.weight'][0])
#conv1_1 = np.array(pixel_cnn.state_dict()['conv_layers.0.weight'])[0]

#plt.imshow(conv1_1.reshape(3, 3), cmap='gray')
#plt.savefig('weight.png')