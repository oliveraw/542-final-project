import config
from utils import torchify, untorchify
from utils.save import save_checkpoint, save_interpolated_videos
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
  

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure
def train_model_distributed(rank, world_size, run_name, B, dataset, outputs_queue):
    torch.cuda.empty_cache()
    print(f"Running basic DDP training on rank {rank}, run name {run_name}")
    setup(rank, world_size)

    if B != None:
        B = B.to(rank)
    model = VideoMLP(B).to(rank)
    with torch.no_grad():                   # dummy forward pass to initialize lazy modules
        (first_xyt, _, _, _, _) = next(iter(dataset))
        first_xyt = first_xyt.to(rank)
        model(first_xyt, 0)

    model = DDP(model, device_ids=[rank])
    model_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_psnrs = []
    # test_psnrs = []
    train_ssims = []
    # test_ssims = []
    for epoch in range(config.ITERATIONS // config.NUM_GPUS + 1):
        IS_MASTER = rank == 0
        RECORD_METRICS = (epoch % config.RECORD_METRICS_INTERVAL == 0) and IS_MASTER
        RECORD_STATE = (epoch % config.RECORD_STATE_INTERVAL == 0) and IS_MASTER

        cur_epoch_train_psnrs = []
        # cur_epoch_test_psnrs = []         # unfortunately did not have the gpu capacity to use 2x resolution for test, can uncomment all these lines if you have the memory
        cur_epoch_train_ssims = []
        # cur_epoch_test_ssims = []
        generated_videos = []
        for data in dataset:
            xyt_train, gt_train, xyt_test, gt_test, data_idx = data
            xyt_train, gt_train= xyt_train.to(rank), gt_train.to(rank)

            optimizer.zero_grad()

            pred_train = model(xyt_train, data_idx).to(rank)
            loss = model_loss(pred_train, gt_train)
            loss.backward()
            optimizer.step()

            if RECORD_METRICS:
                pred_train, gt_train = torchify(pred_train), torchify(gt_train)     # only necessary to be in (-1, C, H, W) format for ssim
                cur_epoch_train_psnrs.append(peak_signal_noise_ratio(pred_train, gt_train).item())
                cur_epoch_train_ssims.append(structural_similarity_index_measure(pred_train, gt_train).item())

                # with torch.no_grad():
                #     xyt_test, gt_test  = xyt_test.to(rank), gt_test.to(rank)
                #     pred_test = model(xyt_test, data_idx)
                #     pred_test, gt_test = torchify(pred_test), torchify(gt_test)     # only necessary to be in (-1, C, H, W) format for ssim
                #     cur_epoch_test_psnrs.append(peak_signal_noise_ratio(pred_test, gt_test).item())
                #     cur_epoch_test_ssims.append(structural_similarity_index_measure(pred_test, gt_test).item())

                if RECORD_STATE:
                    generated_videos.append(untorchify(pred_train))

        if RECORD_METRICS:
            train_psnrs.append(np.average(cur_epoch_train_psnrs))
            # test_psnrs.append(np.average(cur_epoch_test_psnrs))
            train_ssims.append(np.average(cur_epoch_train_ssims))
            # test_ssims.append(np.average(cur_epoch_test_ssims))

            if RECORD_STATE:
                generated_videos = [x.cpu().detach().numpy() for x in generated_videos]
                save_checkpoint(run_name,
                                epoch * config.NUM_GPUS, 
                                generated_videos,
                                model)

    save_interpolated_videos(run_name, model.module, first_xyt)

    # not sure how to handle the metrics collection with distributed, assume just using master is ok
    if IS_MASTER:
        outputs_queue.put({
            'train_psnrs': train_psnrs,
            # 'test_psnrs': test_psnrs,
            'train_ssims': train_ssims,
            # 'test_ssims': test_ssims,
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