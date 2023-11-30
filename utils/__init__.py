from .save import save_checkpoint, save_figs_end

import torch
# changes (1, H, W, 3) img into (1, 3, H, W)
def torchify(x):
    return torch.permute(x, (2, 0, 1))