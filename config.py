import os
import torch 

# dataset related
ROOT_DIR = os.path.join("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsr")

VIDEO_NAME = "water"
VIDEO_DIR = os.path.join(DATA_DIR, f"{VIDEO_NAME}/ground_truth/{VIDEO_NAME}")

FRAME_SUFFIX = ".png"
NUM_FRAMES = 100
# NUM_FRAMES = 2
RESOLUTION = (360, 640)   # this refers to the test resolution, which will be 2x train resolution
TRAIN_RESOLUTION_PIXEL_SKIP = 2
BATCH_SIZE = 1

# model related
NUM_LAYERS = 4
NUM_CHANNELS = 256
USE_PEG = False

# training related
LEARNING_RATE = 1e-4
ITERATIONS = 6001
# ITERATIONS = 1
RECORD_STATE_INTERVAL = ITERATIONS // 6  
RECORD_METRICS_INTERVAL = 25
RECORD_PSNR = True
RECORD_SSIM = False   # idk why just takes too much memory, disable for rn

# positional encoding dim
MAPPING_SIZE = 256

if torch.cuda.is_available():
  DEVICE='cuda'
else:
  DEVICE='cpu'
print(DEVICE)

# save configs
FPS = 30
