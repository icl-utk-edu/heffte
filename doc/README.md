HEFFTE: User manual                
===================

* * *

Installation using CMAKE
========================

To install HEFFTE using CMAKE, create a `build` folder, this will contain object files,
library, and executables.

## Choose 1D FFT backends

A single 1D FFT backend for CPU and GPU can be chosen. Tune options to ON or OFF accordingly to obtain required support.

~~~
mkdir build; cd $_
build/
    cmake -DHeffte_ENABLE_FFTW=ON -DHeffte_ENABLE_CUDA=ON ..
    make -j
~~~

Adding `-DHEFFTE_TIME_DETAILED=true` to CMAKE options, provides an array of
runtime spent per kernel, see `timing.md ` for details.



Installation using Makefile
===========================

We provide two Makefiles for linux systems on folders `src` and `test`, modify them accordingly
to your cluster architecture. Follow command line instructions below to obtain CPU
and GPU versions of HEFFTE.

## Install the GPU enabled HEFFTE
Choose GPU 1D FFT backend, e.g. CUFFT.
~~~
cd heffte/src
src/
    make -j fft=CUFFT
    make install
~~~

Lines above will produce library `libheffte_gpu.a`.

## Install HEFFTE without GPU functionality
Choose CPU 1D FFT backend, e.g. MKL.

~~~
cd heffte/src
src/
    make -j fft=MKL
    make install
~~~

Lines above will produce library `libheffte.a`.

* * *

Running tests
=============

## Verifying correctness

To ensure HEFFTE was properly built, we provide several tests for all kernels. Using CMAKE functionality, simply do as follows:

~~~
cd heffte/build
    ctests -V
~~~

## Running individual tests

If installation was performed using CMAKE, then `heffte/build/test/` will contain several executables; for example, `test3d_cpu` and `test3d_gpu` for C2C FFTs, which can be individually tested.

When using standard Makefile, you can obtain those executables by following lines below:

~~~
cd heffte/test
test/
    make -j fft=FFTW3 tests_cpu
    make -j fft=CUFFT tests_gpu
~~~

To run these tests on an MPI supported cluster, follow the examples:

~~~
mpirun -n 12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c point -s
mpirun -n 12 ./test3d_cpu_r2c -g 512 256 512 -v -i 82783 -c all -pin 1 3 4 -pout 3 4 1 -s
mpirun -n 12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c point -pin 1 3 4 -s
mpirun -n 12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c all -pout 3 4 1 -s
mpirun -n 2  ./test3d_gpu -g 512 256 512 -v -i 82783 -c point -verb -s
mpirun -n 1  ./test3d_gpu_r2c -g 512 256 512 -v -i 82783 -c all -s
~~~

To run on Summit supercomputer, follow the examples:

~~~
jsrun --smpiargs="-gpu" -n192 -a1 -c1 -g1 -r6 ./test3d_gpu -g 1024 1024 1024 -i 82783 -v -c point -s
jsrun  -n2560 -a1 -c1 -r40 ./test3d_cpu -g 1024 1024 1024 -i 82783 -v -c all -s
~~~

* * *

Documentation
=============

To access a detailed documentation on HEFFTE classes and functions you can compile a Doxygen
document by simply typing `make` within this folder.
