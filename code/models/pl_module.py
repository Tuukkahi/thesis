# -*- coding: utf-8 -*-
"""
PyTorch Lightning module for general spatio-temporal optical flow forecasting.
"""
from abc import abstractmethod
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, List
import numpy as np
#from piqa import MS_SSIM

from models.utils import warp, mask
from models.loss import curl, gradient, divergence


class OpticalFlowBase(pl.LightningModule):
  """
  PyTorch Lightning module for spatio temporal forecasting.

  Args:
    model (nn.Module): PyTorch model to use
    **hparams: Hyperparameters for training the model

  Keyword Args:
    lr (float): Learning rate used in the optimizer.
    in_channels (int): Input channels, i.e. lags for the model.
    warping_mode (str): Interpolation mode used for image warping with
                        flow vectors.
    masking (float): Ratio of masked pixels for loss function computation
                     to remove border effects.

  """

  def __init__(self, **hparams):
    super().__init__()
    self.save_hyperparameters()

    self.lr = hparams["lr"]
    self.lags = hparams["in_channels"]
    self.warping_mode = hparams["warping_mode"]
    self.masking = hparams["masking"]
    #self.loss = SSIM(n_channels=1,window_size=19, sigma=4)

    self.training_step_outputs = []
#    self.validation_step_outputs = []
   # self.test_step_outputs = []

    
  @abstractmethod
  def forward(self, x, verbose=False):
    pass

  def single_step(self, x: Tensor, vector_field: Tensor = None) -> Tuple[Tensor, Tensor]:
    """
    Predict next timestep either by propagating a sequence previous images through 
    the network, or using a known vector_field.
    
    Args:
      x (Tensor): Sequence of known images
      vector_field (Tensor): Known vector field for warping. If None, predict a new vector
                             field from the model.
    Returns:
      Tuple[Tensor, Tensor]: Sequence of images that can be used to predict further, and 
                             the used vector field in warping.
    """
    if vector_field is None: 
      vector_field, basis = self(x)
    y_hat = warp(x[:, self.lags - 1], vector_field, self.warping_mode)
    masked_y_hat = mask(y_hat, self.masking)
    #print(vector_field.shape, x.shape, y_hat.shape)
    return torch.cat([x, y_hat], dim=1)[:,1:], vector_field

  def loss_fn(self, prediction: Tensor, target: Tensor, flow: Tensor):
    masked_prediction = mask(prediction, self.masking)
    masked_target = mask(target, self.masking)
    #print(masked_target.shape)
    #return 1 - self.loss(masked_target.unsqueeze(1), masked_prediction.unsqueeze(1))
    return F.mse_loss(masked_target, masked_prediction)
    #return F.mse_loss(masked_target, masked_prediction) + 3000*torch.mean(torch.square(gradient(flow))) +3000*torch.mean(torch.square(divergence(flow))) + 3000*torch.mean(torch.square(curl(flow)))

  def scaler(self, batch: Tensor) -> Tensor:
    maxx = torch.amax(batch, dim=(1,2,3)).reshape(batch.shape[0],1,1,1)
    minn = torch.amin(batch, dim=(1,2,3)).reshape(batch.shape[0],1,1,1)
    batch = (batch - minn)/(maxx - minn)
    return batch

  def training_step(self, batch: Tensor, batch_idx: int) -> Dict:
    """
    Propagate single batch of tensors through the network and compute the
    loss from the model outputs in training.

    Args:
      batch (Tuple[Tensor, Tensor]): A tuple of a batch of image
                                     sequencies x and target images y
      batch_idx (int): batch number

    Returns:
      Dict: Loss from one training step
    """
    batch = torch.rot90(batch, batch_idx % 4, (2,3))
    batch_size = batch.shape[0]
    #batch -= batch.min()
    #batch /= (batch.max()+10*torch.finfo(torch.float32).eps)
    batch = self.scaler(batch)
    
    output = batch[:,0:self.lags]
    flow, basis = self(output)
    loss = 0
    for i in range(batch.shape[1] - self.lags):
      output, flow = self.single_step(output, flow)
      loss += self.loss_fn(output[:,-1], batch[:,i+self.lags], flow)
    loss /= i+1
    self.training_step_outputs.append(loss)
    self.log("train_loss", loss, on_step=True, prog_bar=True)
    return {"loss": loss}

  def training_step_end(self, training_step_outputs: List[Dict]) -> List[Dict]:
    return training_step_outputs

  def on_train_epoch_end(self) -> None:
    """
    Compute average loss from a training step and report it
    """
    epoch_average = torch.stack(self.training_step_outputs).mean()
    self.training_step_outputs.clear()
    self.log("train_loss_epoch", epoch_average)

  def validation_step(self, batch: Tensor,
                      batch_idx: int) -> Dict:
    """
    Propagate a single batch of tensors through the network and compute
    the loss from the model outputs in validation.
    Differs from training_step by not updating model weights.

    Args:
      batch (Tuple[Tensor, Tensor]): A tuple of a batch of image
                                     sequencies x and target images y
      batch_idx (int): batch number

    Returns:
      Dict: Loss from one training step
    """
    batch = torch.rot90(batch, batch_idx % 4, (2,3))
    #batch_size = batch.shape[0]
    #mean = batch.reshape(batch_size, -1).mean(1).view(batch_size,1,1,1)
    #std = batch.reshape(batch_size, -1).std(1).view(batch_size,1,1,1)
    #batch = (batch-mean)/std
    #batch = 5*(torch.tanh(0.01*batch))
    #batch -= batch.min()
    #batch /= (batch.max()+10*torch.finfo(torch.float32).eps)
    batch = self.scaler(batch)
    
    
    output = batch[:,0:self.lags]
    loss = 0
    for i in range(batch.shape[1] - self.lags):
      output, flow = self.single_step(output)
      loss += self.loss_fn(output[:,-1], batch[:,i+self.lags], flow)
      #loss += self.loss_fn(torch.arctanh(1/5*(output[:,-1]))*100, torch.arctanh(1/5*(batch[:,i+self.lags]))*100, flow)
    loss /= i+1
    self.log("val_loss", loss, on_step=True, prog_bar=True)
    return {"loss": loss}


  def predict_step(self, batch: Tensor,
                   batch_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Propagate a batch of tensors through the network to get predictions.

    Output can be used as input
    for this function to get further predictions.
    Args:
      batch (Tensor): Input image sequence for the model.
      batch_idx (int): batch number.

    Returns:
      Tensor: Sequence of images where the last one is prediction.
      Tensor: Predicted flow vector field.
    """
    
    vector_field, basis = self(batch)
    y_hat =  warp(batch[:, -1], vector_field, self.warping_mode)
    return torch.cat([batch, y_hat], dim=1)[:,1:], vector_field, basis

  def propagate(self, x: np.ndarray, scale=True) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(x).unsqueeze(0).float().to(self.device)
    if scale:
      x = self.scaler(x)
    output, flow, _ = self.predict_step(x, 0)
    newx = output[0].cpu().detach().numpy()
    uv = flow[0].cpu().detach().numpy()
    return newx, uv

  def configure_optimizers(self) -> torch.optim.Optimizer:
    """
    Configure optimizer for the model with learning rate
    from the model hyperparameters.

    Returns:
      torch.optim.Optimizer: PyTorch Optimizer instance for training.
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer

