# -*- coding: utf-8 -*-
"""
PyTorch module for a general UNet architecture. 
Constructs a model using parts from `model.unet_parts`.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

from models.unet_parts import DoubleConv, DecoderBlock, EncoderBlock
from models.utils import warp, mask
from models.pl_module import OpticalFlowBase


class UNet(OpticalFlowBase):
  """
  UNet PyTorch model.

  Args:
    **hparams: Hyperparameters for creating a UNet 

  Keyword Args:
    in_channels (int): Input channels, i.e. lags for the model.
    out_channels (int): Number of output channels in the UNet model.
                        Dimension of flow vectors.
    start_channels (int): Number of output channels from the first
                          convolution layer. Affects channel depth for
                          the entire network.
    num_blocks (int): How many encoder and decoder blocks to use
                      in the UNet.
    strided_conv (bool): Whether to apply a strided convolution or
                         maxpool downsamping in encoding phase.
    transposed_conv (bool): Whether to use a transposed convolution or
                            a bilinear upsampling in decoding phase.
    relu_slope (float): Slope for the negative part of leaky relu in
                        activations after convolutions.

  """

  def __init__(self, **hparams):
    super().__init__(**hparams)

    in_channels = hparams["in_channels"]
    start_channels = hparams["start_channels"]
    strided_conv = hparams["strided_conv"]
    num_blocks = hparams["num_blocks"]
    transposed_conv = hparams["transposed_conv"]
    out_channels = hparams["out_channels"]
    relu_slope = hparams["relu_slope"]
    #self.loss = SSIMLoss(data_range=215.2, downsample=False, kernel_size=39, kernel_sigma=20)

    encoder = [
        DoubleConv(in_channels=in_channels,
                   out_channels=start_channels,
                   relu_slope=relu_slope)
    ]
    channels = start_channels
    for _ in range(num_blocks - 1):
      encoder.append(
          EncoderBlock(in_channels=channels,
                       out_channels=channels * 2,
                       strided_conv=strided_conv,
                       relu_slope=relu_slope))
      channels *= 2
    self.encoder = nn.ModuleList(encoder)

    decoder = []
    for _ in range(num_blocks - 1):
      decoder.append(
          DecoderBlock(in_channels=channels,
                       out_channels=channels // 2,
                       transposed_conv=transposed_conv,
                       relu_slope=relu_slope))
      channels //= 2
    self.decoder = nn.ModuleList(decoder)
    
    self.out = nn.Sequential(
      nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
      #nn.Upsample(scale_factor=4, mode="bilinear"),
      #torchvision.transforms.GaussianBlur(kernel_size=65, sigma=30.0)
    )
    
    # Initialize model weights before training closer to the final weights
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.04 / n)
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


  def forward(self, x: Tensor) -> Tensor:
    """
    Moves input tensor x through the entire UNet model.
    Applies first the encoder phase and saves the results for
    skip connections in decoding phase.

    Args:
      x (Tensor): A batch of image sequencies for the model.

    Returns:
      Tensor: A Tensor containing flow vectors found from
      an image sequence.
    """
    encoder_outputs = []
    for block in self.encoder:
      x = block(x)
      encoder_outputs.append(x)
    encoder_outputs.reverse()

    y = encoder_outputs[0]
    for i in range(len(self.decoder)):
        y = self.decoder[i](y, encoder_outputs[i+1])

    return self.out(y), None
