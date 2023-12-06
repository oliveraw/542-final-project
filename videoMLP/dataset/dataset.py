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

        # reuse these xy coords for all images, since all the same size
        self.H, self.W = config.RESOLUTION
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        coord_grid = np.meshgrid(x_coords, y_coords)

        # generate a (100, x, y, t) tensor of xyt inputs for each frame of the video
        # input is the same across all videos
        xyt_for_all_frames = []
        for timestamp in range(config.NUM_FRAMES):
            timestamps = np.ones((self.H, self.W)) * timestamp
            xyt = np.dstack((*coord_grid, timestamps))      # x,y,t (H,W,3) tensor
            if self.transform:
                xyt = self.transform(xyt)
            xyt = torch.Tensor(xyt)
            xyt_for_all_frames.append(xyt)
        self.xyt_for_all_frames = torch.stack(xyt_for_all_frames)

        # assume you can load all videos into memory at the same time (we only use a few, 100 frames each at 360x640)
        video_names = config.VIDEO_NAMES
        video_dirs = [os.path.join(config.DATA_DIR, f"{video_name}/ground_truth/{video_name}") for video_name in video_names]
        self.videos = []
        for video_dir in video_dirs:
            video = []
            frame_names = sorted(os.listdir(video_dir))[:config.NUM_FRAMES]
            for frame_name in frame_names:
                frame_path = os.path.join(video_dir, frame_name)
                frame = read_image(frame_path)
                if self.target_transform:
                    frame = self.target_transform(frame)
                frame = frame.permute(1, 2, 0)/ 255.0 
                video.append(frame)
            video = torch.stack(video)
            self.videos.append(video)


    def __len__(self):
        return config.NUM_VIDEOS

    def __getitem__(self, idx):
        gt_video = self.videos[idx]

        skip = config.TRAIN_RESOLUTION_PIXEL_SKIP
        return self.xyt_for_all_frames[:, ::skip, ::skip, :], gt_video[:, ::skip, ::skip, :], self.xyt_for_all_frames, gt_video, idx
    

def get_dataset():
    return WAIC_TSR()

def get_dataloader():
    dataset = WAIC_TSR()
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)