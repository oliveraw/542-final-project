import config

import torch
import torch.nn as nn

class LatentCodes(nn.Module):
    def __init__(self):
        super(LatentCodes, self).__init__()
        self.latents = nn.parameter.Parameter(
           torch.rand((config.NUM_IMAGES_TO_USE, 1, 1, config.LATENT_DIMENSION), requires_grad=True)
        )

    def forward(self, idx):
        return self.latents[idx]


# input: (224, 224, pe_dim+latent_dim)
# output: (224, 224, 3)
class Latent2DMLP(nn.Module):
    def __init__(self):
      super(Latent2DMLP, self).__init__()

      layers = []
      prev_channels = config.PE_DIMENSION, config.LATENT_DIMENSION
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


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.latents = LatentCodes()
        self.mlp = Latent2DMLP()

    # coords: (224, 224, pe_dim) tensor of positional encodings (same across each image)
    # idx: index of the image, for this we broadcase latent code to (224, 224, latent_dim) and concatenate as input
    # output should be (224, 224, 3) rgb image
    def forward(self, coords, idx):
        code = self.latents(idx)
        code = code.broadcast_to((*config.RESOLUTION, config.LATENT_DIMENSION))
        
        input = torch.concat((coords, code), -1)
        return self.mlp(input)
