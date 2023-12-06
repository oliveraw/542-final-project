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


# input: (224, 224, pe_dim+latent_dim)
# output: (224, 224, 3)
class Latent2DMLP(nn.Module):
    def __init__(self, using_PE):
      super(Latent2DMLP, self).__init__()

      layers = []
      pe_dimension = (2 * config.PE_DIMENSION) if using_PE else 2
      prev_channels = pe_dimension + config.LATENT_DIMENSION
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
    def __init__(self, using_PE):
        super(CombinedModel, self).__init__()
        self.latents = LatentCodes()
        self.mlp = Latent2DMLP(using_PE)

    # pe_coords: (224, 224, 2*pe_dim) tensor of positional encodings (same across each image)
    # idx: index of the image, for this we broadcase latent code to (224, 224, latent_dim) and concatenate as input
    # output should be (224, 224, 3) rgb image
    def forward(self, pe_coords, idx):
        code = self.latents(idx)
        code = code.unsqueeze(1).unsqueeze(1)
        # print('latent shape', code.shape)
        B, H, W, _ = pe_coords.shape
        code = code.broadcast_to((B, H, W, config.LATENT_DIMENSION))
        # print("broadcasted latent code shape", code.shape)
        
        input = torch.concat((pe_coords, code), -1)
        # print("mlp input", input.shape)
        return self.mlp(input)
    
    def interpolate(self, pe_coords, idx1, idx2):
        code1 = self.latents(idx1)
        code2 = self.latents(idx2)
        interpolations = []
        weights = torch.linspace(0, 1, config.NUM_INTERPOLATIONS).to(config.DEVICE)
        for weight in weights:
            interpolated_code = torch.lerp(code1, code2, weight)
            B, H, W, _ = pe_coords.shape
            interpolated_code = interpolated_code.broadcast_to((B, H, W, config.LATENT_DIMENSION))
            input = torch.concat((pe_coords, interpolated_code), -1)
            interpolations.append(self.mlp(input))
        return torch.stack(interpolations), weights
