#!/usr/bin/env python3
# -*- coding: utf-8; -*-

# model trainer

import os
import sys
import argparse
from typing import List

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# from models.unet import UNet
from models.basisfunction import BasisFunctionModel, polynomial_basis
from data.dataloaders import CloudDataModule

def main(argv: List[str]) -> None:
    """
    Trainer
    """ 
  
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", "-d", default=None, type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--output", "-o", default="/tmp/model.ckpt", type=str)
    parser.add_argument("--logdir", default="/tmp/", type=str)
    parser.add_argument("--nologger", action='store_true')
    opts = parser.parse_args(argv)

    if opts.device is not None:
        device = opts.device
    else:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"
    print(f'using {device}')
    
    hparams = {
      "lr": 0.0000000524,
      "in_channels": 3,
      "start_channels": 64,
      "basis_n": 6,
      "coordinate_interval": [-1,1],
      "img_size": 128,
      "basis": polynomial_basis,
      "out_channels": 6,
      "resnet_blocks": [3,3,3,3],
      "warping_mode": "bilinear",
      "masking": 0.05,
      "test_point": 1,
  }

    model = BasisFunctionModel(**hparams)
    #model = UNet(**hparams)
    
    datamodule = CloudDataModule(data_dir=opts.data,
                               lags=3,
                               interval=1,
                               steps=4,
                               batch_size=1,
                               train_val_test_split=[0.5, 1])
    if opts.nologger:
        tb_logger = None
    else:
        tb_logger = TensorBoardLogger(save_dir=opts.logdir)
    trainer = pl.Trainer(
      accelerator=device,
      enable_progress_bar=True,
      enable_checkpointing=not opts.nologger,
      logger=tb_logger,
      callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
      max_epochs=opts.max_epochs)

    trainer.fit(model, datamodule)
    trainer.save_checkpoint(opts.output)

if __name__ == '__main__':    
    main(sys.argv[1:])
