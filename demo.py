import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import optim
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn 
import os

from torchmetrics.image import PeakSignalNoiseRatio  # Updated import for PSNR

def pad_input(input_,img_multiple_of=8):
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        return input_,height,width

def tile_eval(model,input_,tile=128,tile_overlap =32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    restored = E.div_(W)

    restored = torch.clamp(restored, 0, 1)
    return restored

# TTA augmentation functions
def apply_augmentation(image, aug_type):
    """Apply specified augmentation to the input image."""
    if aug_type == 'original':
        return image
    elif aug_type == 'rot90':
        return torch.rot90(image, 1, [2, 3])
    elif aug_type == 'rot180':
        return torch.rot90(image, 2, [2, 3])
    elif aug_type == 'rot270':
        return torch.rot90(image, 3, [2, 3])
    elif aug_type == 'hflip':
        return torch.flip(image, [3])
    elif aug_type == 'vflip':
        return torch.flip(image, [2])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def revert_augmentation(image, aug_type):
    """Revert the augmentation to align with original orientation."""
    if aug_type == 'original':
        return image
    elif aug_type == 'rot90':
        return torch.rot90(image, -1, [2, 3])
    elif aug_type == 'rot180':
        return torch.rot90(image, -2, [2, 3])
    elif aug_type == 'rot270':
        return torch.rot90(image, -3, [2, 3])
    elif aug_type == 'hflip':
        return torch.flip(image, [3])
    elif aug_type == 'vflip':
        return torch.flip(image, [2])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one, 4 for desnow')

    parser.add_argument('--test_path', type=str, default="test/demo/", help='save path of test images, can be directory or an image')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
    parser.add_argument('--tile',type=bool,default=False,help="Set it to use tiling")
    parser.add_argument('--tile_size', type=int, default=128, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--tta', type=bool, default=True, help='Enable Test-Time Augmentation')
    opt = parser.parse_args()


    ckpt_path = "train_ckpt/" + opt.ckpt_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)
    net  = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    net.eval()

    test_set = TestSpecificDataset(opt)
        
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    
    # Define TTA augmentations
    print(opt.tta)
    augmentations = ['original', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip'] if opt.tta else ['original']

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)
            restored_images = []

            for aug_type in augmentations:
                # Apply augmentation
                aug_input = apply_augmentation(degrad_patch, aug_type)

                if opt.tile:
                    print("Using Tiling with TTA")
                    aug_input, h, w = pad_input(aug_input)
                    restored_aug = tile_eval(net, aug_input, tile=opt.tile_size, tile_overlap=opt.tile_overlap)
                    restored_aug = restored_aug[:, :, :h, :w]
                else:
                    restored_aug = net(aug_input)

                # Revert augmentation
                restored_aug = revert_augmentation(restored_aug, aug_type)
                restored_images.append(restored_aug)

            # Average the restored images
            restored = torch.mean(torch.stack(restored_images), dim=0)
            restored = torch.clamp(restored, 0, 1)

            save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')
