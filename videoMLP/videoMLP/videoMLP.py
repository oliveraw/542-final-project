import numpy as np
import config

import torch
import torch.nn as nn

class LatentCodes(nn.Module):
    def __init__(self):
        super(LatentCodes, self).__init__()
        self.latents = nn.parameter.Parameter(
           torch.rand((config.NUM_VIDEOS, config.LATENT_DIMENSION), requires_grad=True)
        )

    def forward(self, idx):
        return self.latents[idx]
    
    def get_codes(self):
        return self.latents
    

def combine_code_and_PE(pe, code):
  # print("combining pe and code", pe.shape, code.shape)
  F, H, W, _ = pe.shape
  code = code.broadcast_to((F, H, W, config.LATENT_DIMENSION))
  return torch.concat((pe, code), -1)

# Fourier feature mapping
# receives (100, H, W, 3) vector
def input_mapping(x, B):
  # print("input mapping, x shape: ", x.shape)
  if B is None:
    return x
  else:
    x_proj = torch.matmul(2.*np.pi*x, B.T)
    res = torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    # print("positional encoded shape", res.shape)
    return res

# in_channels is the only variable layer: depends on positional embedding size
# input size: (H, W, 3)
# output size: (H, W, 3)
class VideoMLP(nn.Module):
    def __init__(self, B):
      super(VideoMLP, self).__init__()

      self.B = B

      if config.USE_LATENTS:
        self.latents = LatentCodes()

      layers = []
      for i in range(config.NUM_LAYERS - 1):
        layers.append(nn.LazyLinear(config.NUM_CHANNELS))
        layers.append(nn.ReLU())
      layers.append(nn.Linear(config.NUM_CHANNELS, 3))
      layers.append(nn.Sigmoid())
      self.hidden_layers = nn.Sequential(*layers)

    # x is directly passed from dataset xyt_for_all_frames
    def forward(self, x, data_idx):
      x = input_mapping(x, self.B)

      if config.USE_LATENTS:
        code = self.latents(data_idx)
        x = combine_code_and_PE(x, code)
      return self.hidden_layers(x)

    def interpolate(self, x, idx1, idx2):
        if not config.USE_LATENTS:
           return [], []
      
        x = input_mapping(x, self.B)
        code1 = self.latents(idx1)
        code2 = self.latents(idx2)
        interpolations = []
        weights = torch.linspace(0, 1, config.NUM_INTERPOLATIONS).to(config.DEVICE)
        for weight in weights:
            interpolated_code = torch.lerp(code1, code2, weight)
            input = combine_code_and_PE(x, interpolated_code)
            # print("after combining latent and pe", input.shape)
            interpolations.append(self.hidden_layers(input))
        return interpolations, weights

