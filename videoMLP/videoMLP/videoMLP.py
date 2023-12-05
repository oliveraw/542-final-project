import config

import torch
import torch.nn as nn

class LatentCodes(nn.Module):
    def __init__(self):
        super(LatentCodes, self).__init__()
        self.latents = nn.parameter.Parameter(
           torch.rand((config.NUM_IMAGES_TO_USE, config.LATENT_DIMENSION), requires_grad=True)
        )

    def forward(self, idx):
        return self.latents[idx]
    
    def get_codes(self):
        return self.latents
    

# in_channels is the only variable layer: depends on positional embedding size
# input size: (H, W, 3)
# output size: (H, W, 3)
class VideoMLP(nn.Module):
    def __init__(self, in_channels):
      super(VideoMLP, self).__init__()

      if config.USE_LATENTS:
        self.latents = LatentCodes()
        prev_channels = in_channels + config.LATENT_DIMENSION
      else:
        prev_channels = in_channels

      layers = []
      for i in range(config.NUM_LAYERS - 1):
        layers.append(nn.Linear(prev_channels, config.NUM_CHANNELS))
        layers.append(nn.ReLU())
        prev_channels = config.NUM_CHANNELS
      layers.append(nn.Linear(config.NUM_CHANNELS, 3))
      layers.append(nn.Sigmoid())
      self.hidden_layers = nn.Sequential(*layers)

    # x is (r x r x mapping_size)
    def forward(self, x, data_idx):
      if config.USE_LATENTS:
        code = self.latents(data_idx)
        code = code.broadcast_to((*config.RESOLUTION, config.LATENT_DIMENSION))
        x = torch.concat((x, code), -1)
      return self.hidden_layers(x)
