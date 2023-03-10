import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F
from torchinfo import summary
from collections import OrderedDict

class Gaze3inputs(nn.Module):
  def __init__(self, block, layers, image_channels, num_bins):
    super(Gaze3inputs, self).__init__()

    self.in_channels = 64
    self.face_res = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
      ('bn1', nn.BatchNorm2d(64)),
      ('relu', nn.ReLU(inplace=True)),
      ('maxppol', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('layer1', self._make_layer(block, layers[0], first_conv_out_channels=64, stride=1)),
      ('layer2', self._make_layer(block, layers[1], first_conv_out_channels=128, stride=2)),
      ('layer3', self._make_layer(block, layers[2], first_conv_out_channels=256, stride=2)),
      ('layer4', self._make_layer(block, layers[3], first_conv_out_channels=512, stride=2)),
      ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    ]))

    self.in_channels = 64
    self.eye_res = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
      ('bn1', nn.BatchNorm2d(64)),
      ('relu', nn.ReLU(inplace=True)),
      ('maxppol', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
      ('layer1', self._make_layer(block, layers[0], first_conv_out_channels=64, stride=1)),
      ('layer2', self._make_layer(block, layers[1], first_conv_out_channels=128, stride=2)),
      ('layer3', self._make_layer(block, layers[2], first_conv_out_channels=256, stride=2)),
      ('layer4', self._make_layer(block, layers[3], first_conv_out_channels=512, stride=2)),
      ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
    ]))

    self.pitch_fc = nn.Linear(512 * block.expansion * 3, num_bins)
    self.yaw_fc = nn.Linear(512 * block.expansion * 3, num_bins)

#init
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


  def forward(self, x1, x2, x3):
    #Get Face Features
    face = self.face_res(x1)
    face = face.view(face.shape[0], -1)

    #Get Eyes Features
    left = self.eye_res(x2)
    left = left.view(left.shape[0], -1)
    right = self.eye_res(x3)
    right = right.view(right.shape[0], -1)

    p_features = torch.cat((face, left, right), 1)
    y_features = torch.cat((face, left, right), 1)

    pre_pitch_gaze = self.pitch_fc(p_features)
    pre_yaw_gaze = self.yaw_fc(y_features)

    return pre_pitch_gaze, pre_yaw_gaze

  def _make_layer(self, block, num_res_blocks, first_conv_out_channels, stride):
    identity_conv = None
    layers = []
    if stride != 1 or self.in_channels != first_conv_out_channels*block.expansion:
      identity_conv = nn.Sequential(
        nn.Conv2d(self.in_channels, first_conv_out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(first_conv_out_channels * block.expansion)
      )

    layers.append(
      block(self.in_channels, first_conv_out_channels, stride, identity_conv)
    )

    self.in_channels = first_conv_out_channels * block.expansion

    for i in range(num_res_blocks - 1):
      layers.append(block(self.in_channels, first_conv_out_channels))

    return nn.Sequential(*layers)

#if you check this network, try to start the code.
#model = GN(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 3, 90)
#summary(model, [(1, 3, 224, 224), (1, 3, 60, 36), (1, 3, 60, 36)])
