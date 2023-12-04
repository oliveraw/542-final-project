import dataset
import config

for B_matrix in config.B_DICT.values():
    dset = dataset.PE_IMAGES(B_matrix)
    print("dataset length", len(dset))

    # data = next(iter(dset))
    pe_coords, img, idx = dset[2]
    print("x, y, shapes", pe_coords.shape, img.shape, "idx:", idx)

    # print(xyt)
    # print(xyt[:,:,2])   # should give timestamp = whatever index of d[.]

    # print(gt_img)

    # hand test: set config.RESOLUTION = (3, 3), make sure xyt and gt_image look right