import os
import torch 

# dataset related
ROOT_DIR = os.path.join("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsr")

VIDEO_NAME = "water"
VIDEO_DIR = os.path.join(DATA_DIR, f"{VIDEO_NAME}/ground_truth/{VIDEO_NAME}")

FRAME_SUFFIX = ".png"
# NUM_FRAMES = 100
NUM_FRAMES = 1
RESOLUTION = (360, 640)   # this refers to the test resolution, which will be 2x train resolution
BATCH_SIZE = 1

# model related
NUM_LAYERS = 4
NUM_CHANNELS = 256
USE_PEG = False

# training related
LEARNING_RATE = 1e-4
# ITERATIONS = 2000
ITERATIONS = 2
RECORD_METRICS_INTERVAL = 25
RECORD_STATE_INTERVAL = 1000

# positional encoding
MAPPING_SIZE = 256

if torch.cuda.is_available():
  print("using GPU")  
  DEVICE='cuda'
else:
  print("using CPU")
  DEVICE='cpu'


# save configs
FPS = 30
