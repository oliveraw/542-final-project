import os
ROOT_DIR = os.path.join("./")
DATA_DIR = os.path.join(ROOT_DIR, "dataset/waic-tsc")
VIDEO_NAME = "water"
VIDEO_DIR = os.path.join(DATA_DIR, f"{VIDEO_NAME}/ground_truth/{VIDEO_NAME}")

FRAME_SUFFIX = ".png"
NUM_FRAMES = 100
RESOLUTION = (256, 452)