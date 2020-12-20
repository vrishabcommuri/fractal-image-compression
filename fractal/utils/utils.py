import numpy as np
from PIL import Image
from collections.abc import MutableSequence


class Partition(MutableSequence):
    def __init__(self, img, mode='equipartition4'):
        if isinstance(img, Image.Image):
            img = np.asarray(img.getdata())
            
        super().__init__()
        
        self.img = img
        self.nrows, self.ncols = img.shape
        
        if self.nrows != self.ncols:
            raise Exception("Image must be square in order to encode")

        if mode == 'equipartition8':
            self.block_size = 8
            self.mode = 'equipartition8'
            
            if self.nrows % self.block_size != 0:
                raise Exception("Equipartition mode requires the image size"\
                                " to be evenly-divisible by the block size.")

        elif mode == 'equipartition4':
            self.block_size = 4
            self.mode = 'equipartition4'
            
            if self.nrows % self.block_size != 0:
                raise Exception("Equipartition mode requires the image size"\
                                " to be evenly-divisible by the block size.")
            
        elif mode == 'equipartition2':
            self.block_size = 2
            self.mode = 'equipartition2'
            
            if self.nrows % self.block_size != 0:
                raise Exception("Equipartition mode requires the image size"\
                                " to be evenly-divisible by the block size.")
            
        else:
            raise Exception("Encoding mode not defined.")  
            
    def _linear_to_tuple(self, idx):
        """Convert linear index to tuple"""
        nchunks_per_row = self.ncols / self.block_size
        nchunks_per_col = self.nrows / self.block_size
        i = int((idx // nchunks_per_row) * self.block_size)
        j = int((idx % nchunks_per_col) * self.block_size)
        return i, j
    
    def __getitem__(self, idx):
        i, j = self._linear_to_tuple(idx)
        return self.img[i:i+self.block_size, j:j+self.block_size]

    def __setitem__(self, idx, data):
        i, j = self._linear_to_tuple(idx)
        self.img[i:i+self.block_size, j:j+self.block_size] = data
        return 
    
    def __delitem__(self, idx):
        # never delete anything, but this method required by MutableSequence
        pass

    def insert(self, idx, data):
        # never insert anything, but this method required by MutableSequence
        pass

    def __len__(self):
        return int((self.nrows/self.block_size) * (self.ncols/self.block_size))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.__len__():
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def image(self):
        return self.img


def init_range_image(nrows, domain_blocks):
    """Initialize a range image from the set of domain blocks"""
    # TODO reinstate error checking
    # if domain_blocks.mode == 'equipartition4' or \
    #    domain_blocks.mode == 'equipartition2':
    ncols = nrows
    return np.zeros([nrows * domain_blocks.block_size, 
                        ncols * domain_blocks.block_size])
    
    # else:
    #     raise Exception("domain block partition mode not supported.")


