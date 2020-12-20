# Overview

For an introduction to fractal compression, as well a general currency on fractal compression techniques, I recommend starting with our final report at ```./doc/final_report.pdf``` which details the motivation and functionality of this codebase. Another excellent resource is Yuval Fisher's book on fractal compression techniques, which is authoritative and quite thorough.

# Codebase

###  <span style="color:grey"> What is the layout of the code?  </span>

The structure of the codebase is that of a 'typical' python project. The ```./bin/``` directory contains scripts that run various experiments. For instance, the ```./bin/main_svd.py``` runs the SVD-based fractal image compression algorithm. Similarly, ```./bin/main_nmf.py``` runs the non-negative matrix factorization fractal compression experiment. There are some miscellaneous routines in this folder as well, such as the ```./bin/cropper.py``` script which simply crops an image to a specified size. 

The ```./fractal/``` directory contains various routines that do the heavy lifting in our framework. In particular, the ```./fractal/coding.py``` file implements the api for encoding (compressing) and decoding (decompressing) images. The ```./fractal/plotting.py``` and ```./fractal/dataloader.py``` files implment some helpful routing for -- as they imply -- plotting and loading images. Within this directory, there is also an important folder ```./fractal/utils``` which implements routines for partitioning images into blocks (```./fractal/utils/utils.py```) and enabling multiprocessor support (```./fractal/utils/multiproc.py```).

The ```./tests/``` directory contains unit tests. We will touch on welcome additions to the codebase shortly, but note for now that this section of the code is currently nonfunctional. Contributions to this part of the code are welcomed.

<br>

###  <span style="color:grey"> Where do I add a new compression algorithm?  </span>

You would add a new algorithm in the ```./fractal/coding.py``` file. Note that your algorithm should conform to a basic ```encode``` and ```decode``` paradigm so as to be consistent with the interfaces for existing algorithms and to simplify testing. 

<br>

###  <span style="color:grey"> Where do I add a new partitioning scheme?  </span>

The partitioning is handled in ```./fractal/utils/utils.py```, specifically by the ```Partition``` class. You can add new partitioning schemes here.

