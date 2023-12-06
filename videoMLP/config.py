########################## CONFIG.PY #################################
import os
import torch 

DEBUG = False

if torch.cuda.is_available():
  DEVICE='cuda'
else:
  DEVICE='cpu'

# distributed training
DISTRIBUTED = False
NUM_GPUS = 2 if DISTRIBUTED else 1    # num gpus
GPU_BACKEND = "gloo"
MASTER_ADDR = "localhost"
MASTER_PORT = "12345"

ROOT_DIR = os.path.join("/home/oliveraw/eecs542/542-final-project/videoMLP")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsr")

VIDEO_NAMES = ["water", "jelly", "billiard", "running_women"]
NUM_VIDEOS = len(VIDEO_NAMES)

OUTPUT_DIR_NAME = f"outputs-debug" if DEBUG else f"outputs"
OUTPUT_DIR = os.path.join(ROOT_DIR, OUTPUT_DIR_NAME)

NUM_FRAMES = 2 if DEBUG else 100
RESOLUTION = (180, 320)           # this refers to the test resolution
TRAIN_RESOLUTION_PIXEL_SKIP = 2   # indicates that test will be 2x train resolution

# model related
USE_LATENTS = True
LATENT_DIMENSION = 128
NUM_LAYERS = 4
NUM_CHANNELS = 256

# training related
LEARNING_RATE = 1e-4
ITERATIONS = 50 if DEBUG else 9000
RECORD_STATE_INTERVAL = 3000
RECORD_METRICS_INTERVAL = 25
RECORD_PSNR = True
RECORD_SSIM = True

# which positional encodings to use   (the "B" matrix from fourier features high dimensional learning paper)
MAPPING_SIZE = 256
# adding mapping matrices into this dict will induce another training run
B_DICT = {}
B_DICT['none'] = None                                                # Standard network - no mapping
# B_DICT['basic'] = torch.eye(3).to(DEVICE)                          # Basic mapping
B_gauss = torch.randn((MAPPING_SIZE, 3), requires_grad=False)        # Three different scales of Gaussian Fourier feature mappings
GAUSSIAN_STDEV_SCALES = [1., 10., 100.]
for scale in GAUSSIAN_STDEV_SCALES:
  B_DICT[f'gauss{scale}'] = B_gauss * scale


# save related
NUM_INTERPOLATIONS = 5
VIDEO_FORMAT = 'GIF'
FPS = 30


########################## CONFIG.PY #################################