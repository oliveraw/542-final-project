import config
from utils import save_checkpoint
from utils import torchify
from videoMLP.videoMLP import VideoMLP

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT

    # initialize the process group
    dist.init_process_group(config.GPU_BACKEND, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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
def train_model_distributed(rank, world_size, run_name, B, dataset, outputs_queue):
    print(f"Running basic DDP training on rank {rank}, run name {run_name}")
    setup(rank, world_size)

    # do this just to get shape
    (first_xyt, _, _, _, _) = next(iter(dataset))
    if B != None:   # None indicates no postional encoding
       B = B.to(rank)
    in_channels = input_mapping(first_xyt.to(rank), B).shape[-1]

    model = VideoMLP(in_channels).to(rank)
    model = DDP(model, device_ids=[rank])
    model_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_psnrs = []
    test_psnrs = []
    train_ssims = []
    test_ssims = []
    all_generated_videos_train = []
    all_generated_videos_test = []
    for i in range(config.ITERATIONS // config.NUM_GPUS + 1):
        IS_MASTER = rank == 0
        RECORD_METRICS = (i % config.RECORD_METRICS_INTERVAL == 0) and IS_MASTER
        RECORD_STATE = (i % config.RECORD_STATE_INTERVAL == 0) and IS_MASTER
        # print("run name", run_name, "gpu rank", rank, "iteration", i)

        generated_video_train = []
        generated_video_test = []
        gt_video_train = []
        gt_video_test = []
        for data in dataset:
            xyt_train, gt_train, xyt_test, gt_test, data_idx = data
            xyt_train, gt_train, xyt_test, gt_test = xyt_train.to(rank), gt_train.to(rank), xyt_test.to(rank), gt_test.to(rank)

            optimizer.zero_grad()

            y_train_pred = model(input_mapping(xyt_train, B), data_idx).to(rank)

            # print("actually predicted", y_train_pred.shape)
            loss = model_loss(y_train_pred, gt_train)
            loss.backward()
            optimizer.step()

            if RECORD_METRICS:
                generated_video_train.append(torchify(y_train_pred))
                gt_video_train.append(torchify(gt_train))

                with torch.no_grad():
                    y_test_pred = model(input_mapping(xyt_test, B), data_idx)
                    generated_video_test.append(torchify(y_test_pred))
                    gt_video_test.append(torchify(gt_test))

        if RECORD_METRICS:
            generated_video_train = torch.stack(generated_video_train)
            generated_video_test = torch.stack(generated_video_test)
            gt_video_train = torch.stack(gt_video_train)
            gt_video_test = torch.stack(gt_video_test)
            
            if config.RECORD_PSNR:
              train_psnrs.append(peak_signal_noise_ratio(generated_video_train, gt_video_train).item())
              test_psnrs.append(peak_signal_noise_ratio(generated_video_test, gt_video_test).item())
            
            if config.RECORD_SSIM:
              train_ssims.append(structural_similarity_index_measure(generated_video_train, gt_video_train).item())
              test_ssims.append(structural_similarity_index_measure(generated_video_test, gt_video_test).item())

            if RECORD_STATE:
              generated_video_train = generated_video_train.cpu().detach().numpy()
              generated_video_test = generated_video_test.cpu().detach().numpy()
              all_generated_videos_train.append(generated_video_train)
              all_generated_videos_test.append(generated_video_test)
              save_checkpoint(run_name,
                              i * config.NUM_GPUS, 
                              generated_video_train, 
                              generated_video_test,
                              model)

    # not sure how to handle the metrics collection with distributed, assume just using master is ok
    if IS_MASTER:
        outputs_queue.put({
            'train_psnrs': train_psnrs,
            'test_psnrs': test_psnrs,
            'train_ssims': train_ssims,
            'test_ssims': test_ssims,
            'pred_train_vids': np.stack(all_generated_videos_train),
            'pred_test_vids': np.stack(all_generated_videos_test),
        })
    cleanup()


def run_distributed_training(run_name, B, dataset):
    world_size = config.NUM_GPUS

    print("run distributed training", run_name)
    manager = mp.Manager()
    outputs_queue = manager.Queue()
    mp.spawn(train_model_distributed,
        args=(world_size, run_name, B, dataset, outputs_queue),
        nprocs=world_size,
        join=True)
    print("finished training for", run_name)      # join=True ensures that whatever happens is after mp.spawn ends
    return outputs_queue.get()