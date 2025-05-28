import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from torchmetrics.image import PeakSignalNoiseRatio  # Updated import for PSNR


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)  # Adjust data_range if needed
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate PSNR
        psnr_value = self.psnr(restored, clean_patch)
        # Log the training PSNR
        self.log("train_psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        psnr_value = self.psnr(restored, clean_patch)
        self.log("val_psnr", psnr_value, on_step=False, on_epoch=True, prog_bar=True)
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=10,max_epochs=100)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")
    
    trainset = PromptTrainDataset(opt, split='train')
    validset = PromptTrainDataset(opt, split='valid')
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    validloader = DataLoader(validset, batch_size=opt.batch_size, pin_memory=True, shuffle=False,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_psnr",
        mode="max",
        dirpath = opt.ckpt_dir,
        filename="best-model-{epoch:02d}-{val_psnr:.2f}",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback],precision='bf16-mixed'
    )
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=validloader)


if __name__ == '__main__':
    main()



