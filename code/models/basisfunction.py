# -*- coding: utf-8 -*-
"""
PyTorch module for a general UNet architecture. 
Constructs a model using parts from `model.unet_parts`.
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Dict, Tuple, List

from models.pl_module import OpticalFlowBase
from models.resnet import ResNet

def polynomial_basis(coordinate: Tensor, basis_n: int) -> Tensor:
  """
  Creates a basis for second order polynomials in shape [7, coordinate_size**2]

  Polynomial basis assumed to be in the form
  .. math::
    \{ 1, x, y, x^2, y^2\}
  
  Args:
    coordinate (Tensor): Coordinate grid for the basis
    basis_n (int): Length of the basis. 7 for second order polynomial.
  """
  squared = coordinate.square()
  prod = coordinate.prod(0).unsqueeze(0)
  prod_squared = prod.square()
  const = torch.ones_like(prod_squared)
#coordinate, prod, squared
  basis = torch.cat([const, coordinate, prod, squared]).flatten(1)
  return basis

def ndgridm(N):
  """Expand index hypercude."""
  N = np.asarray(N)
  # Allocate space for indices
  NN = np.zeros((N.prod(), N.size))
  # For each level/diemsion
  if N.size == 1:
    # The lowest level
    NN[:, 0] = np.arange(1, N + 1)
  else:
    # This level
    n = np.arange(1, N[0] + 1)
    # Recursive call
    nn = ndgridm(N[1:])
    # Assign values
    NN[:, 0] = np.kron(n, np.ones((1, np.prod(N[1:]))))
    NN[:, 1:] = np.tile(nn.ravel(), N[0]).reshape(-1, 1)
  return NN

def eigenval(n):
  """Eigenvalues."""
  return np.sum((np.pi * (n / (2 * np.array([1, 1]))))**2, axis=1)

def makePf(nx, ny, k, indexing='xy'):
  """Generate Fourier basis on [-1,1]*[-1,1]."""
  n = nx * ny
  nn = np.ceil(np.sqrt(n)).astype(int)
  NN = ndgridm([nn, nn])
  eigs = eigenval(NN)
  inds = np.argsort(eigs)
  NN = NN[inds[:k], :]
  eigs = np.sqrt(eigs[inds[:k]])
#    NN = NN[np.argsort(eigs)[:k], :]
#    eigs = np.sqrt(np.sort(eigenval(NN)))

  (xx, yy) = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny),
                         indexing=indexing, sparse=False)
  x = np.c_[xx.ravel(), yy.ravel()]
    
  v = np.empty((n, k))
  for i in range(k):
    v[:, i] = np.prod(np.sin(np.pi * (NN[i, :] * x) / 2), axis=1) / eigs[i]
  return v

def fourier_basis(coordinate: Tensor, basis_n: int) -> Tensor:
  P = makePf(coordinate.shape[1], coordinate.shape[2], basis_n)
  return torch.swapaxes(torch.tensor(P, dtype=torch.float), 0,1)

def test_point(n: int) -> Tensor:
  def intera(x):
    return np.c_[x, x[:,0]*x[:,1], x[:,0]**2, x[:,1]**2]

  V = [[np.cos(phi), np.sin(phi)] for phi in np.linspace(0, 2*np.pi, n)[:-1]]
  V = np.vstack([V, np.zeros((1,2))])
  V = intera(V)
  V = np.c_[np.ones((n,1)), V]
  Vinv = np.linalg.inv(V)
  Vinv[np.abs(Vinv) < 1e-6] = 0.0
  return torch.tensor(Vinv,dtype=torch.float)

class BasisFunctionModel(OpticalFlowBase):
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
    self.basis_n = hparams["basis_n"]
    self.interval = hparams["coordinate_interval"]
    self.img_size = hparams["img_size"]
    self.basis_func = hparams["basis"]
    self.resnet_blocks = hparams["resnet_blocks"]
    self.in_channels = hparams["in_channels"]
    self.is_test_point = hparams["test_point"]
    
    if self.basis_func == polynomial_basis and self.basis_n != 6:
      sys.exit("Polynomial basis only implemented for 6 basis functions")
    if self.basis_func != polynomial_basis and self.is_test_point:
      sys.exit("Test point only implemented for polynomial basis")
    

    coordinates = torch.stack(torch.meshgrid([torch.linspace(self.interval[0],self.interval[1],steps=self.img_size)]*2, indexing='xy'))
    self.register_buffer("basis", self.basis_func(coordinates, self.basis_n))
    if self.is_test_point:
      self.register_buffer("Vinv", test_point(6))

    self.enc_i = ResNet(self.resnet_blocks, in_channels=self.in_channels, num_classes=self.basis_n)

    self.enc_j = ResNet(self.resnet_blocks, in_channels=self.in_channels, num_classes=self.basis_n)

  def forward(self, x: Tensor, verbose=False) -> Tensor:
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
    enc_i_out = self.enc_i(x, verbose=verbose)
    if verbose:
      print("A_i:", A_i.shape)
    enc_j_out = self.enc_j(x, verbose=verbose)
    if verbose:
      print("A_j:", A_j.shape)
    
    if self.is_test_point:
      A_i = torch.matmul(self.Vinv, enc_i_out.unsqueeze(2)).squeeze(-1)
      A_j = torch.matmul(self.Vinv, enc_j_out.unsqueeze(2)).squeeze(-1)
    else:
      A_i = enc_i_out
      A_j = enc_j_out
    F_i = torch.matmul(A_i.unsqueeze(1), self.basis).view(A_i.shape[0], self.img_size, self.img_size)
    F_j = torch.matmul(A_j.unsqueeze(1), self.basis).view(A_j.shape[0], self.img_size, self.img_size)
    return torch.stack([F_i, F_j], dim=1), torch.stack([enc_i_out, enc_j_out], dim=1)
