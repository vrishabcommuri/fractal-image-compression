# Ideas for Futher Research

This document will serve as a sort of inventory of ideas that contributors can undertake to improve the framework. The ideas below are presented in no particular order.

## Sklearn-Style API
The current api is quite loose in the sense that the encoder and decoder are always specified by a unique function. For instance, ```encode_svd(...)``` is distinct from ```encode_nmf(...)```. Ideally, the encoder would be a method bound to a class, so that coding would have a consistent api irrespective of algorithm. For instance, the above would become something like ```SVD.encode(...)``` and ```NMF.encode(...)```. Enforcing adherence to the api in this manner will produce cleaner backend code and a better user-facing interface.

## Better Multiprocessor Support
The current multiprocessor implementation forks a new process for every block of the partitioned image that gets processed. This is a really inefficient implementation, since ```fork``` is quite a heavyweight syscall and to process an individual block only requires the computation of a few parameters (which have closed-form solutions --  a constant time computation). A better approach would be to feed each process several blocks at a time, allowing each process to handle a set of blocks rather than an individual. 

This was the original intent of the multiprocessor code, but I did not have time to refine the implementation. It should function like MapReduce, wherein a subset of blocks are allocated to each processor and then the results are pruned by each until the best block is found for each subset (the Mapping phase). The best blocks are then compared to find the overall best (the Reducing phase).

## Working Test Suite
The test suite is broken and needs to be fixed.



