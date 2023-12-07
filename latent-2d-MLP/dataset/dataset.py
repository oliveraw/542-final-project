import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio

import config

def save_gt_imgs_to_output_dir(filenames):
    src_dir = config.DATA_DIR
    dest_dir = os.path.join(config.OUTPUT_DIR, "gt_images")
    os.makedirs(dest_dir, exist_ok=True)
    for i, filename in enumerate(filenames):
        src_path = os.path.join(src_dir, filename)
        output_path = os.path.join(dest_dir, f"{i}.png")
        shutil.copy(src_path, output_path)
    

class PE_IMAGES(Dataset):
    # Fourier feature mapping
    # x: (H, W, 2) vector of (x, y)
    # B: (pe_dim, 2) matrix for positional encoding
    def input_mapping(self, x, B):
        if B is None:
            return x
        else:
            x_proj = torch.matmul(2.*np.pi*x, B.T)
            res = torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return res
    
    def __init__(self, B_matrix):
        self.resize = transforms.Resize(config.RESOLUTION)
        img_names = sorted(os.listdir(config.DATA_DIR))[0:config.NUM_IMAGES_TO_USE]
        self.img_paths = [os.path.join(config.DATA_DIR, img_name) for img_name in img_names]
        # for img_name in img_names:
        #     img_path = os.path.join(config.DATA_DIR, img_name)
        #     img = imageio.imread(img_path)
        #     print(img_path, img.shape)
        #     img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))      # to torch form (224, 224, 3) -> (3, 224, 224)
        #     img = self.resize(img)
        #     img = img.permute((1, 2, 0))      # undo torch form
        #     img = img / 255.0
        #     self.imgs.append(img)
        
        self.H, self.W = config.RESOLUTION
        x_coords = np.linspace(0, 1, self.W, endpoint=False)   
        y_coords = np.linspace(0, 1, self.H, endpoint=False)    
        coord_grid = torch.tensor(np.stack(np.meshgrid(x_coords, y_coords), -1)).type(torch.float32)
        self.pe_coord_grid = self.input_mapping(coord_grid, B_matrix)

    def __len__(self):
        return config.NUM_IMAGES_TO_USE
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = imageio.imread(img_path)
        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))      # to torch form (224, 224, 3) -> (3, 224, 224)
        img = self.resize(img)
        img = img.permute((1, 2, 0))      # undo torch form
        img = img / 255.0
        return self.pe_coord_grid, img, idx
