from pathlib import Path
import numpy as np
from PIL import Image
from skimage.transform import rescale

class Loader:
    _DEFAULT_DATA_PATH_ROOT = Path(__file__).resolve().parent.parent / "data"
    _DEFAULT_DATA_PATH_FACES = _DEFAULT_DATA_PATH_ROOT / "faces"

    def __init__(self, optpath=None, regex="*.jpg"):
        if optpath is None:
            images = list(self._DEFAULT_DATA_PATH_FACES.glob("*.pgm"))

            # get the first image to learn the image shape
            nrows, ncols = np.asarray(Image.open(images[0])).shape
        
            self._nrows = nrows
            self._ncols = ncols

            self._X = np.zeros([nrows * ncols, len(images)])
            for i, imagepath in enumerate(images):
                image = Image.open(imagepath)
                self._X[:,i] = np.asarray(image.getdata())
        
        else:
            # TODO error checking
            path = self._DEFAULT_DATA_PATH_ROOT / optpath
            images = list(path.glob(regex))

            # get the first image to learn the image shape
            nrows, ncols = np.asarray(Image.open(images[0]).convert('L')).shape
        
            self._nrows = nrows
            self._ncols = ncols

            self._X = np.zeros([nrows * ncols, len(images)])

            for i, imagepath in enumerate(images):
                image = Image.open(imagepath).convert('L')
                imarr = np.asarray(image.getdata())
                self._X[:,i] = imarr


    @property
    def X(self):
        return self._X

    def get_image_PIL(self, i):
        # get the PIL image from the input data
        image = Image.fromarray(np.uint8(self._X[:,0]\
                    .reshape(self._nrows, self._ncols)))

        return image


    def get_image(self, i, scale_factor=1):
        if scale_factor <= 0 or scale_factor > 1:
            raise Exception("Scale factor must be a float between 0 and 1") 

        # get the image from input data
        image = self._X[:,i].reshape(self._nrows, self._ncols)

        # scale it if necessary
        if scale_factor != 1:
            image = rescale(image, scale_factor, anti_aliasing=True)

        return image

    def get_image_1d(self, i):
        return self._X[:,i]
