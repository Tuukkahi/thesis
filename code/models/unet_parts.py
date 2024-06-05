# -*- coding: utf-8 -*-
"""
Torch modules needed for construction of a UNet model.
Includes encoder and decoder modules of a UNet and a double convolutional
module used in the network.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class DoubleConv(nn.Module):
  """
  Double convolutional block module for UNet.
  First convolution layer to update number of channels followed by
  batch normalization and relu.
  Second convolution layer keeps the feature depth and followed by relu.

  Args:
    in_channels (int): Number of input channels of tensor.
    out_channels (int): Number of output channels from the module.
    stride (int): First conv stride for possible downsampling. Default: 1.
    kernel_size (int): Kernel size for convolution layers. Default: 3.
    relu_slope (float): Slope for the negative part of Leaky ReLU.
                        With 0.0 Returns default ReLU. Default = 0.0.
  """

  def __init__(self,
               in_channels: int,
               out_channels: int,
               stride: int = 1,
               kernel_size: int = 3,
               relu_slope: float = 0.0):
    super().__init__()
    self.seq = nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=(kernel_size - 1) // 2,
                  bias=False), nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
        nn.Conv2d(out_channels,
                  out_channels,
                  kernel_size=kernel_size,
                  padding=(kernel_size - 1) // 2,
                  bias=False), nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=relu_slope, inplace=True))

  def forward(self, x: Tensor) -> Tensor:
    """
    Moves input tensor x through the layers.

    Args:
      x (Tensor): Input tensor for the model

    Returns:
      Tensor: Tensor moved through the model
    """
    return self.seq(x)


class EncoderBlock(nn.Module):
  """
  Single encoder block for encoding part in the UNet architecture.
  Either a DoubleConv module with strided convolution to downsample a tensor
  or maxpool layer for downsampling followed by unstrided DoubleConv module.

  Args:
    in_channels (int): Number of input channels of tensor.
    out_channels (int): Number of output channels from the module.
    strided_conv (bool): Whether to apply a strided convolution
                         or maxpooling strategy for downsampling.
    relu_slope (float): Slope for the negative part of Leaky ReLU
                        for DoubleConv module.
  """

  def __init__(self, in_channels: int, out_channels: int, strided_conv: bool,
               relu_slope: float):
    super().__init__()

    if strided_conv:
      self.seq = DoubleConv(in_channels,
                            out_channels,
                            stride=2,
                            relu_slope=relu_slope)
    else:
      self.seq = nn.Sequential(
          nn.MaxPool2d(kernel_size=2, stride=2),
          DoubleConv(in_channels=in_channels, out_channels=out_channels))

  def forward(self, x: Tensor) -> Tensor:
    """
    Moves input tensor x through the layers.

    Args:
      x (Tensor): Input tensor for the model

    Returns:
      Tensor: Tensor moved through the model
    """
    return self.seq(x)


class DecoderBlock(nn.Module):
  """
  Single decoder block for decoding part in the UNet architecture.
  Uses either transposed conv or bilinear upsampling as upsampling strategy.
  Upsamples a tensor and concatenates with another one for skip connection.

  Args:
    in_channels (int): Number of input channels of tensor.
    out_channels (int): Number of output channels from the module.
    transposed_conv (bool): Whether to apply a transposed convolution
                            or bilinear upsampling.
    relu_slope (float): Slope for the negative part of
                        Leaky ReLU in DoubleConv module.
  """

  def __init__(self, in_channels: int, out_channels: int, transposed_conv: bool,
               relu_slope: float):
    super().__init__()

    if transposed_conv:
      self.upsample = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
    else:
      self.upsample = nn.Sequential(
          nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
          nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False))

    self.conv_block = DoubleConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 relu_slope=relu_slope)

  def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
    """
    Moves input tensor x through the layers.

    Args:
      x1 (Tensor): Input tensor for upsampling
      x2 (Tensor): Tensor from encoder for skip connection

    Returns:
      Tensor: Tensor returned from upsampling and skip connection
    """
    x1 = self.upsample(x1)

    # Padding to match dimensions of x1 and x2 needed for concatenation
    diff_x = x2.size()[3] - x1.size()[3]
    diff_y = x2.size()[2] - x1.size()[2]
    x1 = F.pad(
        x1,
        [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    return self.conv_block(torch.cat([x2, x1], dim=1))
