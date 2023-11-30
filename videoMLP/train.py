import config
from utils import save_checkpoint
from utils import torchify
from videoMLP.videoMLP import VideoMLP

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = config.DEVICE

# Fourier feature mapping
# receives (H, W, 3) vector of (x, y, t)
def input_mapping(x, B):
#   print("x shape: ", x.shape)
  if B is None:
    return x
  else:
    x_proj = torch.matmul(2.*np.pi*x, B.T)
    res = torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    # print("positional encoded shape", res.shape)
    return res


from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
# Train model with given hyperparameters and data
def train_model(run_name, B, dataset):

    # do this just to get shape
    (first_xyt, _, _, _) = next(iter(dataset))
    in_channels = input_mapping(first_xyt.to(device), B).shape[-1]

    model = VideoMLP(in_channels).to(device)
    model_loss = nn.MSELoss()
    # model_psnr = PeakSignalNoiseRatio().to(device)
    # model_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    record_iterations = []
    train_psnrs = []
    test_psnrs = []
    train_ssims = []
    test_ssims = []
    all_generated_videos_train = []
    all_generated_videos_test = []
    xs = []
    for i in range(config.ITERATIONS+1):
        RECORD_METRICS = i % config.RECORD_METRICS_INTERVAL == 0
        RECORD_STATE = i % config.RECORD_STATE_INTERVAL == 0

        generated_video_train = []
        generated_video_test = []
        gt_video_train = []
        gt_video_test = []
        for data in dataset:
            xyt_train, gt_train, xyt_test, gt_test = data
            xyt_train, gt_train, xyt_test, gt_test = xyt_train.to(device), gt_train.to(device), xyt_test.to(device), gt_test.to(device)

            optimizer.zero_grad()

            y_train_pred = model(input_mapping(xyt_train, B)).to(device)

            if RECORD_METRICS:
                generated_video_train.append(torchify(y_train_pred))
                gt_video_train.append(torchify(gt_train))

                with torch.no_grad():
                    y_test_pred = model(input_mapping(xyt_test, B))
                    generated_video_test.append(torchify(y_test_pred))
                    gt_video_test.append(torchify(gt_test))

            # print("actually predicted", y_train_pred.shape)
            loss = model_loss(y_train_pred, gt_train)
            loss.backward()
            optimizer.step()

        if RECORD_METRICS:
            generated_video_train = torch.stack(generated_video_train)
            generated_video_test = torch.stack(generated_video_test)
            gt_video_train = torch.stack(gt_video_train)
            gt_video_test = torch.stack(gt_video_test)

            record_iterations.append(i)
            
            if config.RECORD_PSNR:
              # train_psnrs.append(model_psnr(generated_video_train, gt_video_train).item())
              # test_psnrs.append(model_psnr(generated_video_test, gt_video_test).item())
              train_psnrs.append(peak_signal_noise_ratio(generated_video_train, gt_video_train).item())
              test_psnrs.append(peak_signal_noise_ratio(generated_video_test, gt_video_test).item())
            
            if config.RECORD_SSIM:
              # train_ssims.append(model_ssim(generated_video_train, gt_video_train).item())
              # test_ssims.append(model_ssim(generated_video_test, gt_video_test).item())
              train_ssims.append(structural_similarity_index_measure(generated_video_train, gt_video_train).item())
              test_ssims.append(structural_similarity_index_measure(generated_video_test, gt_video_test).item())

            if RECORD_STATE:
              generated_video_train = generated_video_train.cpu().detach().numpy()
              generated_video_test = generated_video_test.cpu().detach().numpy()
              all_generated_videos_train.append(generated_video_train)
              all_generated_videos_test.append(generated_video_test)
              save_checkpoint(run_name,
                              i, 
                              generated_video_train, 
                              generated_video_test,
                              model,
                              record_iterations, 
                              train_psnrs,
                              test_psnrs, 
                              train_ssims,
                              test_ssims)

    return {
        # 'state': get_params(opt_state),
        'train_psnrs': train_psnrs,
        'test_psnrs': test_psnrs,
        'train_ssims': train_ssims,
        'test_ssims': test_ssims,
        'pred_train_vids': np.stack(all_generated_videos_train),
        'pred_test_vids': np.stack(all_generated_videos_test),
        # 'xs': xs,
    }
