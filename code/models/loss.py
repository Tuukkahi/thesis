import torch
from torch import Tensor


def divergence(w: Tensor) -> Tensor:
  """
  Compute divergence of a vector field tensor along it's second dimension.
  
  Args:
    w (Tensor): 4D input tensor
  Returns:
    Tensor: Divergence of w
  """
  dims = w.shape[1]
  return sum([torch.gradient(w[:,i], dim=i+1)[0] for i in range(dims)])

def gradient(w: Tensor) -> Tensor:
  """
  Compute gradient of a vector field tensor along it's second dimension.
  -
  Args:
    w (Tensor): 4D input tensor
  Returns:
    Tensor: Gradient of w
  """
  dims = w.shape[1]
  return torch.stack(torch.gradient(w[:,0], dim=[1,2]) + torch.gradient(w[:,1], dim=[1,2]))

def curl(w: Tensor) -> Tensor:
  return torch.gradient(w[:,1], dim=1)[0] - torch.gradient(w[:,0], dim=2)[0]