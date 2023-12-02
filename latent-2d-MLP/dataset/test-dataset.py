import dataset

dset = dataset.get_dataset()
print("dataset length", len(dset))

# data = next(iter(dset))
img, target = dset[0]
print("first image shape", type(img))

# print(xyt)
# print(xyt[:,:,2])   # should give timestamp = whatever index of d[.]

# print(gt_img)

# hand test: set config.RESOLUTION = (3, 3), make sure xyt and gt_image look right