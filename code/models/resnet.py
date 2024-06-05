# -*- coding: utf-8 -*-
"""
PyTorch ResNet model.
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               3,
                               stride=stride,
                               bias=False,
                               padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               3,
                               bias=False,
                               padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        if stride!=1 or in_channels!=out_channels:
            self.resolution = nn.Sequential(
                            nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=stride,
                                      bias=False),
                            nn.BatchNorm2d(out_channels)
                            )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(self.batchnorm1(y))
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = y + x if (self.stride==1 and self.in_channels==self.out_channels) else y + self.resolution(x)
        return F.relu(y)

class GroupOfBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(GroupOfBlocks, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        other_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]
        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):
        return self.group(x)

class ResNet(nn.Module):
  def __init__(self, n_blocks, in_channels, n_channels=64, num_classes=10):
    """
    Args:
      n_blocks (list):  A list with n elements which contains the number of blocks in 
                        each of the n groups of blocks in ResNet.
                        For instance, n_blocks = [2, 4, 6] means that the first group has two blocks,
                        the second group has four blocks and the third one has six blocks.
      n_channels (int):  Number of channels in the first group of blocks.
      num_classes (int): Number of classes.
    """
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=5, stride=1, padding=2, bias=False)
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    groups = [GroupOfBlocks(n_channels, n_channels, n_blocks[0])]
    for block in n_blocks[1:]:
      groups.append(GroupOfBlocks(n_channels, 2*n_channels, block, stride=2))
      n_channels *= 2
    self.groups = nn.ModuleList(groups)

    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(n_channels, num_classes, bias=False)

    # Initialize weights
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.04 / n)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def forward(self, x, verbose=False):
    """
    Args:
      x of shape (batch_size, 1, ny, nx): Input images.
      verbose: True if you want to print the shapes of the intermediate variables.

    Returns:
      y of shape (batch_size, 10): Outputs of the network.
    """
    if verbose: print(x.shape)
    x = self.conv1(x)
    if verbose: print('conv1:  ', x.shape)
    x = self.bn1(x)
    if verbose: print('bn1:    ', x.shape)
    x = self.relu(x)
    if verbose: print('relu:   ', x.shape)
    x = self.maxpool(x)
    if verbose: print('maxpool:', x.shape)

    for i, group in enumerate(self.groups):
        x = group(x)
        if verbose: print(f"group{i}: {x.shape}")

    x = self.avgpool(x)
    if verbose: print('avgpool:', x.shape)

    x = x.view(-1, self.fc.in_features)
    if verbose: print('x.view: ', x.shape)
    x = self.fc(x)
    if verbose: print('out:    ', x.shape)

    return x
