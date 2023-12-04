import os
import torch 

DEBUG = True

if torch.cuda.is_available():
  DEVICE='cuda'
else:
  DEVICE='cpu'
print(DEVICE)


ROOT_DIR = os.path.join("./")
OUTPUT_DIR_NAME = f"outputs-debug" if DEBUG else f"outputs"
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)

# data
DATA_DIR = os.path.join(ROOT_DIR, "dataset", "rand-images")
RESOLUTION = (224, 224)
NUM_IMAGES_TO_USE = 10

# model and latent
PE_DIMENSION = 128
LATENT_DIMENSION = 128
NUM_LAYERS = 4
NUM_CHANNELS = 512
# SKIP_CONNECTION_INTERVAL =

# adding mapping matrices into this dict will induce another training run
B_DICT = {}
B_DICT['none'] = None                                 # Standard network - no mapping
# B_DICT['basic'] = torch.eye(3).to(DEVICE)             # Basic mapping
B_gauss = torch.randn((PE_DIMENSION, 2)).to(DEVICE)     # Three different scales of Gaussian Fourier feature mappings
GAUSSIAN_STDEV_SCALES = [1., 10., 100.]
for scale in GAUSSIAN_STDEV_SCALES:
  B_DICT[f'gauss{scale}'] = B_gauss * scale 

# training related
LEARNING_RATE = 1e-4
ITERATIONS = 1 if DEBUG else 3001
RECORD_STATE_INTERVAL = 1000
RECORD_METRICS_INTERVAL = 25
RECORD_PSNR = True
RECORD_SSIM = True


##########################################################