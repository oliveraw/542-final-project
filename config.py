import os
import torch 

DEBUG = True

if torch.cuda.is_available():
  DEVICE='cuda'
else:
  DEVICE='cpu'
print(DEVICE)

ROOT_DIR = os.path.join("./")
OUTPUT_DIR_NAME = "outputs-debug" if DEBUG else "outputs"
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)
NPY_DIR = os.path.join(OUTPUT_DIR, "npy")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsr")

VIDEO_NAME = "water"
VIDEO_DIR = os.path.join(DATA_DIR, f"{VIDEO_NAME}/ground_truth/{VIDEO_NAME}")

FRAME_SUFFIX = ".png"
NUM_DIGITS_FOR_PNG_NAME = '05'
NUM_FRAMES = 2 if DEBUG else 100
RESOLUTION = (360, 640)   # this refers to the test resolution, which will be 2x train resolution
TRAIN_RESOLUTION_PIXEL_SKIP = 2
BATCH_SIZE = 1

# model related
NUM_LAYERS = 4
NUM_CHANNELS = 256
USE_PEG = False

# training related
LEARNING_RATE = 1e-4
ITERATIONS = 1 if DEBUG else 6001
RECORD_STATE_INTERVAL = 1000
RECORD_METRICS_INTERVAL = 25
RECORD_PSNR = True
RECORD_SSIM = True

# which positional encodings to use   (the "B" matrix from fourier features high dimensional learning paper)
MAPPING_SIZE = 256
# adding mapping matrices into this dict will induce another training run
B_DICT = {}
# B_DICT['none'] = None                                 # Standard network - no mapping
# B_DICT['basic'] = torch.eye(3).to(DEVICE)             # Basic mapping
B_gauss = torch.randn((MAPPING_SIZE, 3)).to(DEVICE)     # Three different scales of Gaussian Fourier feature mappings
GAUSSIAN_STDEV_SCALES = [10., 100.]
for scale in GAUSSIAN_STDEV_SCALES:
  B_DICT[f'gauss{scale}'] = B_gauss * scale

