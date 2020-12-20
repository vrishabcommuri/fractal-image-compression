from context import fractal
from fractal import utils
import numpy as np


if __name__ == '__main__':
    X = np.eye(20)
    xp = utils.Partition(X, 'equipartition4')
    for i in xp:
        print(xp)
        print(xp.shape)
        print()