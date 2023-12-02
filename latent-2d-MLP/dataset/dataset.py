import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

import config

# only use cifar 10 train set, don't care about labels or train/test just need images
def get_dataset():
    return torchvision.datasets.CIFAR10(config.DATA_DIR, 
                                        download=True, 
                                        train=True, 
                                        transform=config.TRANSFORM,
                                        target_transform=config.TARGET_TRANSFORM)

def get_dataloader():
    dataset = get_dataset()
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)