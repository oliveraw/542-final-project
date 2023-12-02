import config

import torch
import torch.nn as nn

# in_channels is the only variable layer: depends on positional embedding size
# input size: (H, W, 3)
# output size: (H, W, 3)
class VideoMLP(nn.Module):
    def __init__(self, in_channels):
      super(VideoMLP, self).__init__()

      self.use_PEG = config.USE_PEG

      if self.use_PEG:
        self.PEG = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding="same")

      layers = []
      prev_channels = in_channels
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
      if self.use_PEG:
        x = torch.permute(x, (2, 0, 1))   # move size d to first axis: (mapping_size x r x r)
        x = self.PEG(x) + x
        x = torch.permute(x, (1, 2, 0))   # revert back to (r, r, mapping_size)

      # print("after PEG conv", x.shape)
      return self.hidden_layers(x)
