import imageio
import os

import config
import dataset

d = dataset.WAIC_TSR()

assert len(d) == config.NUM_VIDEOS
# print(d.xyt_for_all_frames[:, 10, 10, :])

# xyt_half, gt_img_half, xyt, gt_img, idx = d[0]

# print("xyt_half shape:", xyt_half.shape, "gt_img_half shape", gt_img_half.shape)
# print("xyt shape:", xyt.shape, "gt_img shape:", gt_img.shape)
# print("video index", idx)

# print(xyt)
# print(xyt[:,:,2])   # should give timestamp = whatever index of d[.]

# print(gt_img)

# hand test: set config.RESOLUTION = (3, 3), make sure xyt and gt_image look right

# save dataset videos in resized shape
for data in d:
    _, gt_vid_half, _, _, idx = data
    print("gt_vid_half shape", gt_vid_half.shape)
    gif_path = os.path.join(config.OUTPUT_DIR, "gt", f"gt_{idx}.gif")
    imageio.mimsave(gif_path, gt_vid_half, format=config.VIDEO_FORMAT, fps=config.FPS)