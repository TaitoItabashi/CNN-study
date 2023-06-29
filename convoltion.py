"""任意の画像を使って畳み込みをするプログラム"""

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import conv2d
from PIL import Image

"""weight_visualizationで可視化した重み"""
weight = torch.tensor([[[
        [ 0.0683, -0.1079,  0.0516,  0.0324, -0.0638,  0.0898, -0.0004],
        [ 0.0209, -0.0386, -0.0699,  0.0495, -0.0119, -0.0380, -0.1065],
        [-0.1312,  0.0934, -0.1072, -0.1483,  0.1732,  0.0296, -0.1295],
        [ 0.0914,  0.0544, -0.1688, -0.0000, -0.0000, -0.0000, -0.0000],
        [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
        [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]]], dtype=torch.float)

"""縦方向の特徴抽出"""
vertical = torch.tensor([[[
        [-2, 1, 1],
        [-2, 1, 1],
        [-2, 1, 1]]]], dtype=torch.float)

"""横方向の特徴抽出"""
horizontal = torch.tensor([[[
        [1, 1, 1],
        [1, 1, 1],
        [-2, -2, -2]]]], dtype=torch.float)

image = torch.from_numpy(np.array(Image.open('./image/camera.png')).astype(np.float32))
image_ = image.reshape(1, image.size()[0], image.size()[1])
print("Image size:", image.size(), "weight size:",weight.size())

conv_image = torch.reshape(conv2d(image_, horizontal, padding=1), (image.size()[0], image.size()[1]))
print(conv_image.size())

Image.fromarray(conv_image.to('cpu').detach().numpy().astype(np.uint8)).save('./image/conv_camera_h.png')
