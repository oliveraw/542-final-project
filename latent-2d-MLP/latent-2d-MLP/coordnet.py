import config

import torch
import torch.nn as nn

class CoordNet(nn.Module):
    def __init__(self, in_channels):
      super(CoordNet, self).__init__()

      layers = []
      prev_channels = config.LATENT_DIMENSION + 2       # input: 128(latent) + 2(xy)
      for i in range(config.NUM_LAYERS - 1):
        layers.append(nn.Linear(prev_channels, config.NUM_CHANNELS))
        layers.append(nn.ReLU())
        prev_channels = config.NUM_CHANNELS
      layers.append(nn.Linear(config.NUM_CHANNELS, 3))
      layers.append(nn.Sigmoid())
      self.hidden_layers = nn.Sequential(*layers)

    # x is (r x r x mapping_size)
    def forward(self, x):
      # print("input shape", x.shape)
      return self.hidden_layers(x)
