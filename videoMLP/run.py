import dataset
import config
from videoMLP.train import train_model

import torch

waic_dataset = dataset.get_dataset()

B_dict = {}

# Standard network - no mapping
B_dict['none'] = None

# Basic mapping
# B_dict['basic'] = torch.eye(2).to(device=device)

# Three different scales of Gaussian Fourier feature mappings
B_gauss = torch.randn((config.MAPPING_SIZE, 3)).to(device=config.DEVICE)
for scale in [1., 10., 100.]:
  B_dict[f'gauss_{scale}'] = B_gauss * scale

# This should take about 2-3 minutes
outputs = {}
for k in B_dict:
  print("starting training for", k)
  outputs[k] = {}
  outputs[k]['standard'] = train_model(k, B_dict[k], waic_dataset)
  # outputs[k]['PEG'] = train_model(network_size, learning_rate, iters, B_dict[k], train_data, test_data, True)