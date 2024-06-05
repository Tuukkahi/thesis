import os
import sys
import torch
import numpy as np
import xarray as xr
import numpy.typing as npt
import pytorch_lightning as pl
from glob import glob
from typing import List


class SequenceDataset(torch.utils.data.Dataset):

  def __init__(self, X: npt.NDArray, lags: int, steps: int):
    self.X = X
    self.lags = lags
    self.steps = steps
    self.valid_indices = dict()

  def __len__(self) -> int:
    index = 0
    valid_seq_i = 0
    while index < (self.X.__len__() - (self.lags + self.steps)):
      X = self.X[index:index + self.lags + self.steps]
      if np.sum(np.isnan(X)) == 0:
        self.valid_indices[valid_seq_i] = index
        valid_seq_i += 1
      index += 1
    self.length = valid_seq_i
    return valid_seq_i

  def __getitem__(self, index: int) -> torch.Tensor:
    i = self.valid_indices[index]
    X = self.X[i:i + self.lags + self.steps]
    if np.sum(np.isnan(X)) != 0:
      sys.exit("NAN")
    return X


class SSTDataModule(pl.LightningDataModule):

  def __init__(self,
               data_dir: str,
               batch_size: int = 16,
               train_val_test_split: List[float] = [0.7, 0.9],
               lags: int = 3,
               interval: int = 1):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.split = train_val_test_split
    self.lags = lags
    self.interval = interval

  def setup(self, stage=None):
    images = np.load(self.data_dir)
    images = images[:, ::self.interval]
    self.train_images = images[:, :int(images.shape[1] * self.split[0])]
    self.val_images = images[:,
                             int(images.shape[1] *
                                 self.split[0]):int(images.shape[1] *
                                                    self.split[1])]
    self.test_images = images[:,
                              int(images.shape[1] *
                                  self.split[1]):images.shape[1]]

  def train_dataloader(self):
    trainsets = []
    for i in range(self.train_images.shape[0]):
      trainsets.append(
          SequenceDataset(self.train_images[i], self.lags, self.steps))
    trainset = torch.utils.data.ConcatDataset(trainsets)
    return torch.utils.data.DataLoader(trainset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True)

  def val_dataloader(self):
    valsets = []
    for i in range(self.val_images.shape[0]):
      valsets.append(SequenceDataset(self.val_images[i], self.lags, self.steps))
    valset = torch.utils.data.ConcatDataset(valsets)
    return torch.utils.data.DataLoader(valset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=2)

  def test_dataloader(self):
    testsets = []
    for i in range(self.test_images.shape[0]):
      testsets.append(
          SequenceDataset(self.test_images[i], self.lags, self.steps))
    testset = torch.utils.data.ConcatDataset(testsets)
    return torch.utils.data.DataLoader(testset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=2)

  def predict_dataloader(self):
    testsets = []
    for i in range(self.test_images.shape[0]):
      testsets.append(
          SequenceDataset(self.test_images[i], self.lags, self.steps))
    testset = torch.utils.data.ConcatDataset(testsets)
    return torch.utils.data.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=2)


class CloudDataModule(pl.LightningDataModule):

  def __init__(self,
               data_dir: str,
               batch_size: int = 16,
               train_val_test_split: List[float] = [0.7, 0.9],
               lags: int = 3,
               steps: int = 1,
               interval: int = 1):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.split = train_val_test_split
    self.lags = lags
    self.steps = steps
    self.interval = interval

  def setup(self, stage=None):
    ds = xr.open_dataset(self.data_dir)
    self.days = ds.groupby("timei")
    self.train_range = range(0, int(self.split[0] * len(self.days)))
    self.val_range = range(int(self.split[0] * len(self.days)), int(self.split[1] * len(self.days)))
    self.test_range = range(int(self.split[1] * len(self.days)), len(self.days))

  def train_dataloader(self):
    trainsets = []
    for day_i in self.train_range:
      try:
        for region in self.days[day_i].imgs.data:
          trainsets.append(SequenceDataset(region[::self.interval], self.lags, self.steps))
      except KeyError:
        continue
    trainset = torch.utils.data.ConcatDataset(trainsets)
    return torch.utils.data.DataLoader(trainset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True)

  def val_dataloader(self):
    valsets = []
    for day_i in self.val_range:
      try:
        for region in self.days[day_i].imgs.data:
          valsets.append(SequenceDataset(region[::self.interval], self.lags, self.steps))
      except KeyError:
        continue
    valset = torch.utils.data.ConcatDataset(valsets)
    return torch.utils.data.DataLoader(valset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       num_workers=2)

  def test_dataloader(self):
    testsets = []
    for day_i in self.test_range:
      try:
        for region in self.days[day_i].imgs.data:
          testsets.append(SequenceDataset(region[::self.interval], self.lags, self.steps))
      except:
        continue
    testset = torch.utils.data.ConcatDataset(testsets)
    return torch.utils.data.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=2)
