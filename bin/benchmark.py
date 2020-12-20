from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode_svd, decode_svd
from fractal.coding import encode, decode
import copy
from skimage.transform import rescale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time

MODE = 'equipartition4'
SCALE = 0.25
BLOCK_SIZE = 16
COMPRESSION_RATIO = int(0.75*BLOCK_SIZE) # 75% compression 


def load_aaron_face():
    facedir = Path(__file__).resolve().parent.parent / "data" / "faces"
    aaronface = Image.open(list(facedir.glob("*.pgm"))[0])
    aaronface = np.asarray(aaronface.getdata()).reshape(64,64)
    return aaronface


if __name__ == '__main__':
    ############################################################################
    # load images
    ############################################################################
    images = Loader("")
    range_image = images.get_image(0, scale_factor=SCALE)

    # plot it so we know what it looks like
    plot_image(range_image, 
        title=f"Range Image {range_image.shape[0]}x{range_image.shape[1]}",
        cmap='gray')

    ############################################################################
    # divide up the first image into chunks
    ############################################################################
    # domain image is a 50% downsampled range image
    domain_image = images.get_image(0, scale_factor=SCALE/2)

    plot_image(domain_image, 
        title=f"Domain Image {domain_image.shape[0]}x{domain_image.shape[1]}",
        cmap='gray')

    # each block is 4x4
    domain_chunks = utils.Partition(domain_image, mode=MODE)
    range_chunks = utils.Partition(range_image, mode=MODE)

    ############################################################################
    # encode the range image using standard encoding 
    ############################################################################
    
    # encode image
    start = time.time()
    codebook = encode(domain_chunks, range_chunks, multiproc=False, verbose=False)
    standard_encoding_time = time.time() - start

    # load aaron's face to use as input for the reconstruction
    aaronface = load_aaron_face()
    plot_image(aaronface, 
        title=f"Reconstruction Input {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')
    domain_chunks = utils.Partition(aaronface, mode=MODE)

    # decode image 100 times
    start = time.time()
    reconstructed_chunks = decode(codebook, domain_chunks)
    for i in range(99):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode(codebook, domain_chunks)
    
    standard_decoding_time = time.time()-start

    plot_image(reconstructed_chunks.image, 
        title=f"Reconstructed Image (Standard Encoding) \n {reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]}", 
        cmap='gray', y=0.97)

    ############################################################################
    # encode the range image using parallelized standard encoding 
    ############################################################################
    
    # encode image
    domain_chunks = utils.Partition(domain_image, mode=MODE)

    start = time.time()
    codebook = encode(domain_chunks, range_chunks, multiproc=True, verbose=False)
    multiproc_encoding_time = time.time() - start

    # load aaron's face to use as input for the reconstruction
    aaronface = load_aaron_face()
    plot_image(aaronface, 
        title=f"Reconstruction Input {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')
    domain_chunks = utils.Partition(aaronface, mode=MODE)

    # decode image 100 times
    start = time.time()
    reconstructed_chunks = decode(codebook, domain_chunks)
    for i in range(99):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode(codebook, domain_chunks)
    
    multiproc_decoding_time = time.time()-start

    plot_image(reconstructed_chunks.image, 
        title=f"Reconstructed Image (Standard Parallel Encoding) \n {reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]}", 
        cmap='gray', y=0.97)

    ############################################################################
    # encode the range image using svd encoding 
    ############################################################################
    
    # encode image
    domain_chunks = utils.Partition(domain_image, mode=MODE)
    start = time.time()
    codebook = encode_svd(domain_chunks, range_chunks, verbose=False)
    svd_encoding_time = time.time() - start

    # for svd only we implement compression by 
    # dropping coefficients in the codebook
    codebook[:, COMPRESSION_RATIO:] = 0

    # load aaron's face to use as input for the reconstruction
    aaronface = load_aaron_face()
    plot_image(aaronface, 
        title=f"Reconstruction Input {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray', y=0.97)
    domain_chunks = utils.Partition(aaronface, mode=MODE)

    # decode image 100 times
    start = time.time()
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    for i in range(99):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
    
    svd_decoding_time = time.time()-start

    plot_image(reconstructed_chunks.image, 
        title=f"Reconstructed Image (SVD Encoding) \n {reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]}", 
        cmap='gray', y=0.97)


    ############################################################################
    # encoding and decoding time results
    ############################################################################

    print(f"standard mode encoding: {standard_encoding_time}")
    print(f"standard mode decoding: {standard_decoding_time}")
    print(f"parallel mode encoding: {multiproc_encoding_time}")
    print(f"parallel mode decoding: {multiproc_decoding_time}")
    print(f"svd mode encoding: {svd_encoding_time}")
    print(f"svd mode decoding: {svd_decoding_time}")

