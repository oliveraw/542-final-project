import config
from utils import save_checkpoint
from utils import torchify
from latent2dMLP.latent2dMLP import CombinedModel

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.io as io
import numpy as np
import os

device = config.DEVICE

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
# Train model with given hyperparameters and data
def train_model(run_name, B, dataset):

    using_PE = (B != None)

    model = CombinedModel(using_PE).to(device)
    model_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    psnrs = []
    ssims = []
    generated_images = []
    gt_images = []
    for i in range(config.ITERATIONS):
        RECORD_METRICS = i % config.RECORD_METRICS_INTERVAL == 0
        RECORD_STATE = i % config.RECORD_STATE_INTERVAL == 0

        for data in dataset:
            pe_coords, img, idx = data
            pe_coords, img = pe_coords.to(device), img.to(device)

            optimizer.zero_grad()

            pred = model(pe_coords, idx).to(device)

            loss = model_loss(pred, img)
            loss.backward()
            optimizer.step()

            if RECORD_METRICS:
                generated_images.append(pred)
                gt_images.append(img)

        if RECORD_METRICS:
            generated_images = torch.stack(generated_images)
            gt_images = torch.stack(gt_images)
            
            if config.RECORD_PSNR:
              psnrs.append(peak_signal_noise_ratio(generated_images, gt_images).item())
            
            if config.RECORD_SSIM:
              ssims.append(structural_similarity_index_measure(generated_images, gt_images).item())

            # if RECORD_STATE:
            #   generated_video_train = generated_video_train.cpu().detach().numpy()
            #   generated_video_test = generated_video_test.cpu().detach().numpy()
            #   all_generated_videos_train.append(generated_video_train)
            #   all_generated_videos_test.append(generated_video_test)
            #   save_checkpoint(run_name,
            #                   i, 
            #                   generated_video_train, 
            #                   generated_video_test,
            #                   model)

    run_final_inference(run_name, dataset, model)

    return {
        'psnrs': psnrs,
        'ssims': ssims,
        # 'generated_images': torch.stack(generated_images),
        # 'gt_images': torch.stack(gt_images),
    }

def run_final_inference(run_name, dataset, model):
  output_dir = os.path.join(config.OUTPUT_DIR, run_name)
  os.makedirs(output_dir, exist_ok=True)

  for data in dataset:
     pe_coords, img, idx = data
     with torch.no_grad():
        pred = model(pe_coords, idx)

        pred_path = os.path.join(output_dir, f"{idx}_pred.png")
        gt_path = os.path.join(output_dir, f"{idx}_gt.png")
        io.write_png(pred, pred_path)