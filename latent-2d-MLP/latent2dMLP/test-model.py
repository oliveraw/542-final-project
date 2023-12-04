import dataset
import config
from latent2dMLP.latent2dMLP import CombinedModel, LatentCodes

codes = LatentCodes()
assert codes(1) != codes(2)

for B_matrix in config.B_DICT.values():
    dset = dataset.PE_IMAGES(B_matrix)
    idx = 2
    pe_coords, img, _ = dset[idx]
    print("x, y, shapes", pe_coords.shape, img.shape)

    using_PE = B_matrix != None
    model = CombinedModel(using_PE=using_PE)
    pred = model(pe_coords=pe_coords, idx=idx)
    print("model output shape", pred.shape)