# README #

## HEFFTE

HEFFTE is a new massively parallel FFT library for CPU/GPU architectures.
It is specifically designed with the goal of achieving maximum performance with a good scalability.

### Installation ###
#### Using CMAKE
  * Create a folder *mkdir build*,
  * Then *cd build*,
  * Load a CMAKE version greater than 2.8, *load module cmake*
  * To install the library with complete GPU functionality, type:
    * cmake -DFFTW_ROOT="/ccs/home/fftw-3.3.8" -DFFTW_USE_STATIC_LIBS=true -DBUILD_GPU=true -DCXX_FLAGS="-O3" -DBUILD_SHARED=false ..
  * To install only the CPU library, type
    * cmake -DFFTW_ROOT="/ccs/home/fftw-3.3.8" -DFFTW_USE_STATIC_LIBS=true -DBUILD_GPU=false -DCXX_FLAGS="-O3" -DBUILD_SHARED=false ..

### Using Makefile
  * Modify the Makefile provided and then type *make*

### Documentation ###
A detailed step by step documentation on how to use the library
can be found in the *doc* folder of this repository.

  * Original research paper introducing HEFFTE library:
Markup :   [HEFFTE technical report](https://www.icl.utk.edu/publications/fft-ecp-fast-fourier-transform)

### License ###
HEFFTE is distributed under The University of Tennessee license.
Please see LICENSE file.
