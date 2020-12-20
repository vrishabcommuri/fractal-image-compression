from context import fractal
from fractal.dataloader import Loader
from fractal import utils
from fractal.plotting import plot_image
from fractal.coding import encode_svd, decode_svd
from skimage.transform import rescale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
import os

MODE = 'equipartition8'

def save_contractiveness_factors(domain_chunks, iternum):
    num_weights = len(domain_chunks[0].ravel())
    X = np.zeros([num_weights, len(domain_chunks)])

    for i in range(len(domain_chunks)):
        X[:, i] = np.ravel(domain_chunks[i])

    u, s, vh = np.linalg.svd(X)

    R = np.dot(np.diag(s), vh[:len(s), :len(s)])

    df = None
    for idx, weights in enumerate(codebook):
        print(f'saving contractiveness factors for block {idx}/{len(codebook)}, iteration {iternum}')
        contractiveness = np.dot(np.linalg.inv(R), weights.reshape(num_weights,1))

        contractiveness = np.array([idx] + \
                            contractiveness.ravel().tolist())[np.newaxis, :]
        if df is None:
            df = pd.DataFrame(contractiveness)
        else:
            df = df.append(pd.DataFrame(contractiveness))

    df.to_csv(f"contractiveness_{iternum}.csv")

parser = argparse.ArgumentParser(description='select image')
parser.add_argument('--imagepath', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    ############################################################################
    # load the first face image
    ############################################################################
    images = Loader("", regex=args.imagepath)
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
    print("partitioned")

    ############################################################################
    # encode the range image
    ############################################################################
    codebook = encode_svd(domain_chunks, range_chunks, verbose=False)
    # codebook[:, 8:] = 0 # uncomment to compress representation

    # facedir = Path(__file__).resolve().parent.parent / "data" / "faces"
    if args.imagepath == 'mandrill.jpg':
        facedir = Path(__file__).resolve().parent.parent / "data"
        aaronface = Image.open(list(facedir.glob("lena_gray.jpg"))[0])
        print(aaronface)
        aaronface = np.asarray(aaronface.getdata()).reshape(512,512)
        aaronface = rescale(aaronface, 0.25)
    else:
        facedir = Path(__file__).resolve().parent.parent / "data"
        aaronface = Image.open(list(facedir.glob("mandrill.jpg"))[0])
        aaronface = np.asarray(aaronface.getdata()).reshape(512,512)
        aaronface = rescale(aaronface, 0.25)
    # aaronface = np.asarray(aaronface.getdata()).reshape(64,64)

    # aaronface = images.get_image(0, scale_factor=0.125/2)
    plot_image(aaronface, 
        title=f"Domain Image {aaronface.shape[0]}x{aaronface.shape[1]}",
        cmap='gray')

    domain_chunks = utils.Partition(aaronface, mode=MODE)
    reconstructed_chunks = decode_svd(codebook, domain_chunks)

    # save_contractiveness_factors(domain_chunks, 1)
    
    for i in range(26):
        rec_dim = rescale(reconstructed_chunks.image, 0.5) 
        domain_chunks = utils.Partition(rec_dim, mode=MODE)
        # save_contractiveness_factors(domain_chunks, i)
        reconstructed_chunks = decode_svd(codebook, domain_chunks)
        if i in [1, 5, 10, 20, 25]:
            plot_image(reconstructed_chunks.image, 
            title=f"Reconstructed Image {i} iterations \n{reconstructed_chunks.\
                image.shape[0]}x{reconstructed_chunks.image.shape[0]} ({MODE})", 
            cmap='gray', y=0.98)
            print(f"residual: {i},  {np.sum((range_image -\
                reconstructed_chunks.image)**2)}")


    # pd.DataFrame(reconstructed_chunks.image).to_csv("reconstruction.csv")


