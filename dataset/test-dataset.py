import dataset

d = dataset.WAIC_TSC()
xyt, gt_img = d[9]

print("input shape:", xyt.shape, "img shape:", gt_img.shape)

# print(xyt[:,:,2])   # should give timestamp = whatever index of d[.]

# print(gt_img)