import numpy as np
import config
from utils import save_final_checkpoint
from latent2dMLP.latent2dMLP import CombinedModel
import dataset

import torch
import torch.nn as nn
import torch.optim as optim

device = config.DEVICE

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
# Train model with given hyperparameters and data
def train_model(run_name, B):
    dset = dataset.PE_IMAGES(B)
    dloader = torch.utils.data.DataLoader(dset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)

    using_PE = (B != None)

    model = CombinedModel(using_PE).to(device)
    model_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    psnrs = []
    ssims = []
    for epoch in range(config.ITERATIONS):
        RECORD_METRICS = epoch % config.RECORD_METRICS_INTERVAL == 0

        psnr_per_epoch = []
        ssim_per_epoch = []
        for data in dloader:
            pe_coords, gt_imgs, idx = data
            pe_coords, gt_imgs = pe_coords.to(device), gt_imgs.to(device)

            pred = model(pe_coords, idx).to(device)
            loss = model_loss(pred, gt_imgs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if RECORD_METRICS:
                pred = pred.permute((0, 3, 1, 2))    # convert to (B, C, H, W) format
                gt_imgs = gt_imgs.permute((0, 3, 1, 2))
                if config.RECORD_PSNR:
                    psnr_per_epoch.append(peak_signal_noise_ratio(pred, gt_imgs).item())
                
                if config.RECORD_SSIM:
                    ssim_per_epoch.append(structural_similarity_index_measure(pred, gt_imgs).item())
                
        if RECORD_METRICS:
            print("recording metric, epoch", epoch, "run name", run_name)
            psnrs.append(np.average(psnr_per_epoch))
            ssims.append(np.average(ssim_per_epoch))

    # save final image predictions, latent codes, and model .pth
    save_final_checkpoint(run_name, dloader, model, B)

    return {
        'psnrs': psnrs,
        'ssims': ssims,
    }