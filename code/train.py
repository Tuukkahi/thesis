import argparse
import sys
import torch
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

#from models.pl_module import OpticalFlowBase
from models.unet import UNet
from models.basisfunction import BasisFunctionModel, polynomial_basis, fourier_basis
from data.dataloaders import SSTDataModule, CloudDataModule

def main(argv: List[str]) -> None:
  """
  hparams = {
      "lr": 1.3182567385564074e-5,
      "in_channels": 3,
      "start_channels": 64,
      "strided_conv": False,
      "num_blocks": 7,
      "transposed_conv": False,
      "out_channels": 2,
      "warping_mode": "bilinear",
      "relu_slope": 0.01,
      "masking": 0.05,
      "img_size": 128
  }

  """ 
  hparams = {
      "lr": 3e-8,
      "in_channels": 3,
      "start_channels": 64,
      "basis_n": 6,
      "coordinate_interval": [-1.0,1.0],
      "img_size": 128,
      "basis": polynomial_basis,
      "out_channels": 6,
      "resnet_blocks": [3,3,3,3,3],
      "warping_mode": "bilinear",
      "masking": 0.15,
      "test_point": True,
  }

  model = BasisFunctionModel(**hparams)
  #CKPT_PATH = '/scratch/project_2001027/adafume/deepide/lightning_logs/version_169/checkpoints/epoch=8-step=1980.ckpt'
  #checkpoint = torch.load(CKPT_PATH)
  #model = BasisFunctionModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)
  #model = UNet(**hparams)
    
  datamodule = CloudDataModule(data_dir="/scratch/project_2001027/adafume/deepide/data/highres/S_NWC_CMIC_COT_2021-04-01_2021-09-29_cubic_resampling_utm_finland_NN_input_" + str(hparams["img_size"]) + "_unnorm.nc",
                               lags=3,
                               interval=1,
                               steps=4,
                               batch_size=64,
                               train_val_test_split=[0.8,0.9])
  tb_logger = TensorBoardLogger(save_dir="/scratch/project_2001027/adafume/deepide/")
  checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
  trainer = pl.Trainer(
      accelerator='gpu',
      enable_progress_bar=True,
      enable_checkpointing=True,
      logger=tb_logger,
      callbacks=[checkpoint_callback],
      max_epochs=250)

  trainer.fit(model, datamodule)


if __name__ == '__main__':
  main(sys.argv[1:])
