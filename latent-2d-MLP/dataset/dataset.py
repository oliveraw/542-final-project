import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio

import config

class PE_IMAGES(Dataset):
    # Fourier feature mapping
    # x: (H, W, 2) vector of (x, y)
    # B: (pe_dim, 2) matrix for positional encoding
    def input_mapping(self, x, B):
        if B is None:
            return x
        else:
            print(x.dtype, B.dtype)
            x_proj = torch.matmul(2.*np.pi*x, B.T)
            res = torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return res
    
    def __init__(self, B_matrix):
        # self.img_dataset = torchvision.datasets.
        self.filenames = os.listdir(config.DATA_DIR)
        # print(self.filenames)
        
        self.H, self.W = config.RESOLUTION
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        coord_grid = torch.tensor(np.stack(np.meshgrid(x_coords, y_coords), -1)).type(torch.float32)
        self.pe_coord_grid = self.input_mapping(coord_grid, B_matrix)

        self.resize = transforms.Resize(config.RESOLUTION)

    def __len__(self):
        return config.NUM_IMAGES_TO_USE
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(config.DATA_DIR, img_name)
        img = imageio.imread(img_path)
        print("img size", img.shape)
        img = torch.tensor(img).permute((2, 0, 1))      # to torch form (224, 224, 3) -> (3, 224, 224)
        img = self.resize(img)
        img = torch.tensor(img).permute((1, 2, 0))      # undo torch form

        return self.pe_coord_grid, img, idx
