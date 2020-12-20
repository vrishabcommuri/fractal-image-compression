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
SCALE = 1
BLOCK_SIZE = 16
# percentage*block_size determines how many coefficients to retain
# this is where the compression happens. For example, 0.75*BLOCK_SIZE
# will drop 25% of coefficients. Similarly, 0.33*BLOCK_SIZE will zero 
# out 67% (rounded to the nearest integer) coefficients
COMPRESSION_FACTOR = int((1/16)*BLOCK_SIZE) 

def PSNR(original, compressed): 
    from math import log10, sqrt 

    # sometimes recovered signal has out-of-bound values
    compressed = np.clip(compressed, 0, 255)
    
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def load_reconstruction_input():
    facedir = Path(__file__).resolve().parent.parent / "data"
    face = Image.open(list(facedir.glob("mandrill.jpg"))[0])
    face = np.asarray(face.getdata()).reshape(512,512)
    face = rescale(face, 0.25)
    return face


if __name__ == '__main__':
    ############################################################################
    # load images
    ############################################################################
    images = Loader(optpath="", regex="lena.jpg")
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
    # encode the range image using svd encoding 
    ############################################################################
    
    # encode image
    start = time.time()
    codebook = encode_svd(domain_chunks, range_chunks, verbose=False)
    svd_encoding_time = time.time() - start

    # for svd only we implement compression by 
    # dropping coefficients in the codebook
    codebook[:, COMPRESSION_FACTOR:] = 0

    # load an input face to use as input for the reconstruction
    inpface = load_reconstruction_input()
    plot_image(inpface, 
        title=f"Reconstruction Input {inpface.shape[0]}x{inpface.shape[1]}",
        cmap='gray', y=0.97)
    domain_chunks = utils.Partition(inpface, mode=MODE)

    # decode image 100 times
    start = time.time()
    reconstructed_chunks = decode_svd(codebook, domain_chunks)
    
    for i in range(50):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
    
    svd_decoding_time = time.time()-start

    plot_image(reconstructed_chunks.image, 
        title=f"Reconstructed Image (SVD Encoding) \n {reconstructed_chunks.image.shape[0]}x{reconstructed_chunks.image.shape[0]}, 50 iterations", 
        cmap='gray', y=0.97)

    ############################################################################
    # encoding and decoding time results
    ############################################################################

    print(f"svd mode encoding: {svd_encoding_time}")
    print(f"svd mode decoding: {svd_decoding_time}")

    ############################################################################
    # psnr results
    ############################################################################

    print(f"PSNR: {PSNR(range_image, reconstructed_chunks.image)} \t Coefficients Retained: {COMPRESSION_FACTOR}/{BLOCK_SIZE}")
