from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode_nmf, decode_nmf
import copy
from skimage.transform import rescale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

MODE = 'equipartition4'

if __name__ == '__main__':
    ############################################################################
    # load the first face image
    ############################################################################
    images = Loader("")
    range_image = images.get_image(0, scale_factor=1)

    # plot it so we know what it looks like
    plot_image(range_image, 
        title=f"Range Image {range_image.shape[0]}x{range_image.shape[1]}",
        cmap='gray')

    ############################################################################
    # divide up the first image into chunks
    ############################################################################
    # domain image is a 50% downsampled range image
    domain_image = images.get_image(0, scale_factor=1/2)

    plot_image(domain_image, 
        title=f"Domain Image {domain_image.shape[0]}x{domain_image.shape[1]}",
        cmap='gray')

    # each block is 4x4
    domain_chunks = utils.Partition(domain_image, mode=MODE)
    range_chunks = utils.Partition(range_image, mode=MODE)

    ############################################################################
    # encode the range image
    ############################################################################
    codebook = encode_nmf(domain_chunks, range_chunks, verbose=True)
    print(codebook)
    codebook[:, 8:] = 0

    # num_weights = len(domain_chunks[0].ravel())
    # X = np.zeros([num_weights, len(domain_chunks)])

    # for i in range(len(domain_chunks)):
    #     X[:, i] = np.ravel(domain_chunks[i])

    # u, s, vh = np.linalg.svd(X)

    # print(u.shape, np.diag(s).shape, vh.shape)
    # R = np.dot(np.diag(s), vh[:len(s), :len(s)])

    # df = None
    # print(R.shape)
    # for idx, weights in enumerate(codebook):
    #     print(f'finished {idx}/{len(codebook)}')
    #     contractiveness = np.dot(np.linalg.inv(R), weights.reshape(num_weights,1))
    #     contractiveness = np.array([idx] + contractiveness.ravel().tolist())[np.newaxis, :]
    #     if df is None:
    #         df = pd.DataFrame(contractiveness)
    #     else:
    #         df = df.append(pd.DataFrame(contractiveness))

    # df.to_csv("contractiveness.csv")

    facedir = Path(__file__).resolve().parent.parent / "data" / "faces"
    aaronface = Image.open(list(facedir.glob("*.pgm"))[0])
    aaronface = np.asarray(aaronface.getdata()).reshape(64,64)

    # aaronface = images.get_image(0, scale_factor=0.125/2)
    plot_image(aaronface, 
        title=f"Domain Image {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')

    domain_chunks = utils.Partition(aaronface, mode=MODE)
    # domain_chunks = utils.Partition(np.zeros([domain_image.shape[0], 
    #                                         domain_image.shape[1]]), mode=MODE)
    reconstructed_chunks = decode_nmf(codebook, domain_chunks)
    reconstructed_chunks_1iter = copy.deepcopy(reconstructed_chunks.image)
    
    for i in range(10):
        # TODO try this without anti aliasing
        rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_nmf(codebook, domain_chunks)
        plot_image(reconstructed_chunks.image, 
        title=f"Reconstructed Image {i} iteration \n{reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]} ({MODE})", 
        cmap='gray')
        print(f"residual: {i},  {np.sum((range_image - reconstructed_chunks.image)**2)}")
    

    pd.DataFrame(reconstructed_chunks.image).to_csv("reconstruction.csv")
    # reconstructed_chunks_10iter = copy.deepcopy(reconstructed_chunks.image)


    # for i in range(10):
    #     # TODO try this without anti aliasing
    #     rec_dim = rescale(reconstructed_chunks.image, 0.5) #, anti_aliasing=True)
    #     domain_chunks = utils.Partition(rec_dim, mode=MODE)
    #     reconstructed_chunks = decode_svd(codebook, domain_chunks)

    # reconstructed_chunks_100iter = copy.deepcopy(reconstructed_chunks.image)

    # size = reconstructed_chunks_1iter.shape[0]

    # plot the result
    # plot_image(reconstructed_chunks_1iter, 
    #     title=f"Reconstructed Image 1 iteration \n{size}x{size} ({MODE})", 
    #     cmap='gray')
    # plot_image(reconstructed_chunks_10iter, 
    #     title=f"Reconstructed Image 5 iterations \n{size}x{size} ({MODE})",
    #     cmap='gray')
    # plot_image(reconstructed_chunks_100iter, 
    #     title=f"Reconstructed Image 15 iterations \n{size}x{size} ({MODE})",
    #     cmap='gray')





    # print(len(domain_chunks))
    # X = np.zeros([16, 64])

    # for i in range(len(domain_chunks)):
    #     X[:, i] = np.ravel(domain_chunks[i])

    # u, s, vh = np.linalg.svd(X)

    # print(u.shape)
    # O = np.zeros([16+3, 16+3]) - 0.5

    # i = 0
    # for row in range(0, 16+4, 5):
    #     for col in range(0, 16+4, 5):
    #         eigenblock = u[:,i]
    #         e = eigenblock.reshape(4, 4)
    #         print(e)
    #         O[row:row+4, col:col+4] = e
    #         i += 1
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(O, cmap='gray') #, vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()






