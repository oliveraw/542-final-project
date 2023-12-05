import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize

import config

# all outputs as pytorch tensors
# train images at half resolution as test images
class WAIC_TSR(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = None
        self.target_transform = Resize(config.RESOLUTION)

        self.H, self.W = config.RESOLUTION
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        self.coord_grid = np.meshgrid(x_coords, y_coords)

        frame_names = sorted(os.listdir(config.VIDEO_DIR))[:config.NUM_FRAMES]
        self.frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(config.VIDEO_DIR, frame_name)
            frame = read_image(frame_path)
            # need to do the resize before permute
            if self.target_transform:
                frame = self.target_transform(frame)
            frame = frame.permute(1, 2, 0)
            frame = frame / 255.0       # normalize pixels to [0,1]
            self.frames.append(frame)


    def __len__(self):
        return config.NUM_FRAMES

    def __getitem__(self, idx):
        timestamp = idx
        timestamps = np.ones((self.H, self.W)) * timestamp
        xyt = np.dstack((*self.coord_grid, timestamps))      # x,y,t (H,W,3) tensor
        if self.transform:
            xyt = self.transform(xyt)
        xyt = torch.Tensor(xyt)

        gt_image = self.frames[idx]

        skip = config.TRAIN_RESOLUTION_PIXEL_SKIP
        xyt_half = xyt[::skip, ::skip, :]
        gt_image_half = gt_image[::skip, ::skip, :]

        return xyt_half, gt_image_half, xyt, gt_image, idx

def get_dataset():
    return WAIC_TSR()

def get_dataloader():
    dataset = WAIC_TSR()
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)