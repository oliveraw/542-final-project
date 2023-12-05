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

    using_PE = (B != None)

    model = CombinedModel(using_PE).to(device)
    model_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    psnrs = []
    ssims = []
    for epoch in range(config.ITERATIONS):
        RECORD_METRICS = epoch % config.RECORD_METRICS_INTERVAL == 0
        RECORD_STATE = epoch % config.RECORD_STATE_INTERVAL == 0

        generated_images = []
        gt_images = []
        for data in dset:
            pe_coords, img, idx = data
            pe_coords, img = pe_coords.to(device), img.to(device)

            pred = model(pe_coords, idx).to(device)
            loss = model_loss(pred, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if RECORD_METRICS:
                generated_images.append(pred)
                gt_images.append(img)

        if RECORD_METRICS:
            print("recording metric, epoch", epoch, "run name", run_name)
            generated_images = torch.stack(generated_images).permute((0, 3, 1, 2))    # convert to (B, C, H, W) format
            gt_images = torch.stack(gt_images).permute((0, 3, 1, 2))
            # print("generated_images shape", generated_images.shape, "gt_images", gt_images.shape)
            
            if config.RECORD_PSNR:
              psnrs.append(peak_signal_noise_ratio(generated_images, gt_images).item())
            
            if config.RECORD_SSIM:
              ssims.append(structural_similarity_index_measure(generated_images, gt_images).item())

    # save final image predictions, latent codes, and model .pth
    save_final_checkpoint(run_name, dset, model, B)

    return {
        'psnrs': psnrs,
        'ssims': ssims,
    }