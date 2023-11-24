import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

import config

class WAIC_TSC(Dataset):
    def __init__(self, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        # self.transform = transform
        # self.target_transform = target_transform

        self.H, self.W = config.RESOLUTION    # 256 x 412
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        self.coord_grid = np.meshgrid(x_coords, y_coords)

        self.frame_names = sorted(os.listdir(config.VIDEO_DIR))[:config.NUM_FRAMES]

        self.transform = None
        self.target_transform = Resize(config.RESOLUTION)


    def __len__(self):
        return config.NUM_FRAMES

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label

        frame_name = self.frame_names[idx]
        frame_path = os.path.join(config.VIDEO_DIR, frame_name)

        timestamp = int(frame_name.replace(config.FRAME_SUFFIX, ""))
        timestamps = np.ones((self.H, self.W)) * timestamp
        xyt = np.dstack((*self.coord_grid, timestamps))      # x,y,t (H,W,3) tensor
        if self.transform:
            xyt = self.transform(xyt)

        gt_image = read_image(frame_path)
        # need to do the resize before permute
        if self.target_transform:
            gt_image = self.target_transform(gt_image)
        gt_image = gt_image.permute(1, 2, 0)
        gt_image = gt_image / 255.0       # normalize pixels to [0,1]

        return xyt, gt_image