########################## CONFIG.PY #################################
import os
import torch 

DEBUG = False

if torch.cuda.is_available():
  DEVICE='cuda'
else:
  DEVICE='cpu'

# distributed training
DISTRIBUTED = True
NUM_GPUS = 2 if DISTRIBUTED else 1    # num gpus
GPU_BACKEND = "gloo"
MASTER_ADDR = "localhost"
MASTER_PORT = "12345"

ROOT_DIR = os.path.join("/home/oliveraw/eecs542/542-final-project/videoMLP")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsr")

VIDEO_NAME = "water"
VIDEO_DIR = os.path.join(DATA_DIR, f"{VIDEO_NAME}/ground_truth/{VIDEO_NAME}")

OUTPUT_DIR_NAME = f"outputs-debug-{VIDEO_NAME}" if DEBUG else f"outputs-{VIDEO_NAME}"
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)
NPY_DIR = os.path.join(OUTPUT_DIR, "npy")

FRAME_SUFFIX = ".png"
NUM_DIGITS_FOR_PNG_NAME = '05'
NUM_FRAMES = 2 if DEBUG else 100
RESOLUTION = (360, 640)           # this refers to the test resolution
TRAIN_RESOLUTION_PIXEL_SKIP = 2   # indicates that test will be 2x train resolution
BATCH_SIZE = 1

# model related
NUM_LAYERS = 4
NUM_CHANNELS = 256
USE_PEG = False

# training related
LEARNING_RATE = 1e-4
ITERATIONS = 50 if DEBUG else 3000
RECORD_STATE_INTERVAL = 500
RECORD_METRICS_INTERVAL = 25
RECORD_PSNR = True
RECORD_SSIM = True

# which positional encodings to use   (the "B" matrix from fourier features high dimensional learning paper)
MAPPING_SIZE = 256
# adding mapping matrices into this dict will induce another training run
B_DICT = {}
# B_DICT['none'] = None                                 # Standard network - no mapping
# B_DICT['basic'] = torch.eye(3).to(DEVICE)           # Basic mapping
B_gauss = torch.randn((MAPPING_SIZE, 3))              # Three different scales of Gaussian Fourier feature mappings
GAUSSIAN_STDEV_SCALES = [100.]
for scale in GAUSSIAN_STDEV_SCALES:
  B_DICT[f'gauss{scale}'] = B_gauss * scale


# save related
VIDEO_FORMAT = 'GIF'
FPS = 30


########################## CONFIG.PY #################################