import config
import dataset

d = dataset.WAIC_TSR()

assert len(d) == config.NUM_FRAMES

xyt_half, gt_img_half, xyt, gt_img = d[9]

print("xyt_half shape:", xyt_half.shape, "gt_img_half shape", gt_img_half.shape)
print("xyt shape:", xyt.shape, "gt_img shape:", gt_img.shape)

# print(xyt)
# print(xyt[:,:,2])   # should give timestamp = whatever index of d[.]

# print(gt_img)

# hand test: set config.RESOLUTION = (3, 3), make sure xyt and gt_image look right