# -*- coding: utf-8 -*-
"""
Util functions needed in for the UNet model.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def mask(tensor: Tensor, masking_rate: float) -> Tensor:
  """
  Remove values from the last two dimensions of a tensor.

  Args:
   tensor (Tensor): Tensor to mask.
   masking_rate (float): Ratio of pixels from the tensor height
                         and width to remove.

  Returns:
    Tensor: Masked tensor
  """
  pad_px_1 = -int(masking_rate * tensor.shape[-1])
  pad_px_2 = -int(masking_rate * tensor.shape[-2])
  #return tensor[..., -pad_px_2:tensor.shape[-2]+pad_px_2, -pad_px_1:tensor.shape[-1]+pad_px_1]
  return F.pad(tensor, (pad_px_1, pad_px_1, pad_px_2, pad_px_2))


def warp(image: Tensor, vector_field: Tensor, mode: str) -> Tensor:
  """
  Warping an image with a flow vector field using
  torch.nn.functional.grid_sample function.
  The torch function required a vector flow grid normalized to [-1, 1]
  See details:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

  Args:
    image (Tensor): tensor move forward.
    vector_field (Tensor): flow vectors used to move image.

  Returns:
    Tensor: Warped tensor
  """
  d = torch.linspace(-1, 1, vector_field.shape[2], device=vector_field.device)
  meshx, meshy = torch.meshgrid((d, d), indexing='ij')
  grid = torch.stack((meshy, meshx), -1)
  grid = grid.unsqueeze(0)
  vector_field = 2 * vector_field.permute(0, 2, 3, 1) / (vector_field.shape[2] - 1)
  flow_grid = grid - vector_field
  return F.grid_sample(image.unsqueeze(1),
                       flow_grid,
                       mode=mode,
                       padding_mode="border",
                       align_corners=True)
