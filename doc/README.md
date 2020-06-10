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

## Performance evaluation and comparison

Once HEFFTE is built, several tests are available in folder `heffte/build/test/`. These tests allow to evaluate correctness and performance, and they should be used to validate new developments. 

To evaluate scalability and make performance comparison with other parallel FFT libraries, refer to folder `heffte/build/benchmarks/`, where you will find two executables: `speed3d_c2c` for complex-complex transforms, and `speed3d_r2c` for real-to-complex transforms. To run these tests on an MPI supported cluster, follow the examples:

~~~
mpirun -n 12 ./speed3d_r2c fftw double 512 256 512 -p2p -pencils -no-reorder
mpirun -n 5 --map-by node  ./speed3d_c2c mkl single 1024 256 512  -a2a -slabs -reorder
mpirun -n 2 ./speed3d_c2c cufft double 512 256 512  -mps -a2a
~~~

Should you have questions about the use of flags, please refer to `flags.md` for detailed information. For systems, such as Summit supercomputer, which support execution with `jsrun` by default, follow the examples:

~~~
jsrun  -n1280 -a1 -c1 -r40 ./speed3d_r2c fftw double 1024 256 512 -pencils 
jsrun --smpiargs="-gpu" -n192 -a1 -c1 -g1 -r6 ./speed3d_c2c cufft double 1024 1024 1024 -p2p -reorder
~~~

For comparison to other libraries, make sure to use equivalent flags. Some libraries only provide benchmarks for evaluating FFT performance starting and ending at a pencils-shaped FFT grids. For such cases, use the flag `-io_pencils`.

We have kept old benchmark testers from version 0.2, which can be found in folder `test/`, these benchmarks had limited features and can still be tested as showed below.

~~~
mpirun -n 12 ./speed3d_c2c -g 512 256 512 -v -i 82783 -c point -s
mpirun -n 12 ./test3d_cpu_r2c -g 512 256 512 -v -i 82783 -c all -pin 1 3 4 -pout 3 4 1 -s
mpirun -n 1  ./test3d_gpu_r2c -g 512 256 512 -v -i 82783 -c all -s
jsrun --smpiargs="-gpu" -n192 -a1 -c1 -g1 -r6 ./test3d_gpu -g 1024 1024 1024 -i 82783 -v -c point -s
jsrun  -n2560 -a1 -c1 -r40 ./test3d_cpu -g 1024 1024 1024 -i 82783 -v -c all -s
~~~

* * *

Documentation
=============

To access a detailed documentation on HEFFTE classes and functions you can compile a Doxygen
document by simply typing `make` within this folder.
