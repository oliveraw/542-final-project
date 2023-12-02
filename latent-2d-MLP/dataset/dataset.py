import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import config

class PE_CIFAR(Dataset):
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
        self.img_dataset = torchvision.datasets.
        
        self.H, self.W = config.RESOLUTION
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        coord_grid = torch.tensor(np.stack(np.meshgrid(x_coords, y_coords), -1)).type(torch.float32)
        self.pe_coord_grid = self.input_mapping(coord_grid, B_matrix)

    def __len__(self):
        return config.NUM_IMAGES_TO_USE
    
    def __getitem__(self, idx):
        img, _ = self.img_dataset[idx]      # don't need labels of cifar
        img = img.permute((1, 2, 0))
        return self.pe_coord_grid, img
