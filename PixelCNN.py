# referance: https://data-analytics.fun/2021/10/27/understanding-pixelcnn/

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchinfo import summary


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())
        h = self.weight.size()[2]
        w = self.weight.size()[3]
        self.mask.fill_(1)
        # マスクタイプによる場合分け
        if mask_type == 'A': # 自分自身も見ない
          self.mask[:, :, h // 2, w // 2:] = 0
          self.mask[:, :, h // 2 + 1:] = 0
        else: # 自分自身は見る
          self.mask[:, :, h // 2, w // 2 + 1:] = 0
          self.mask[:, :, h // 2 + 1:] = 0
 
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    

class PixelCNN(nn.Module):
    def __init__(self, num_of_channels=32, num_of_layers=7, output_channels=256):
        super(PixelCNN, self).__init__()
        self.layers = nn.ModuleList()
 
        # 最初のブロック
        self.layers.append(MaskedConv2d(mask_type='A',
                                        in_channels=1, 
                                        out_channels=num_of_channels,
                                        kernel_size=7,
                                        stride=1, 
                                        padding=3, 
                                        bias=False))
        self.layers.append(nn.BatchNorm2d(num_of_channels))
        self.layers.append(nn.ReLU(inplace=True))
 
        # 後続のブロック
        for i in range(1, num_of_layers+1):
            self.layers.append(MaskedConv2d(mask_type='B',
                                            in_channels=num_of_channels, 
                                            out_channels=num_of_channels,
                                            kernel_size=7,
                                            stride=1, 
                                            padding=3, 
                                            bias=False))
            self.layers.append(nn.BatchNorm2d(num_of_channels))
            self.layers.append(nn.ReLU(inplace=True))
 
        self.layers.append(nn.Conv2d(in_channels=num_of_channels, 
                                     out_channels=output_channels,
                                     kernel_size=1))
     
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out