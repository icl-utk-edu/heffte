HEFFTE: User manual                
===================

* * *

Installation using CMAKE
========================

To install HEFFTE using CMAKE we create a folder `build`, where we store the object files,
library, and executables.

## Install the GPU enabled HEFFTE

~~~
mkdir build; cd $_
build/
    cmake -DBUILD_GPU=true -DCXX_FLAGS="-O3" -DBUILD_SHARED=false ..
    make -j
~~~

## Install HEFFTE without GPU functionality

This will need to specify the 1DFFT library which by default is FFTW3, simply add the paths to the
library or load the library, e.g. "`module load fftw3`". If not module is available, you can add it
manually with `-DFFTW_ROOT` option.

~~~
build/
    cmake -DFFTW_ROOT="/ccs/home/aayala/fftw-3.3.8" -DBUILD_GPU=false -DCXX_FLAGS="-O3" -DBUILD_SHARED=false ..
    make -j
~~~


Installation using Makefile
===========================

We provide two makefiles for linux systems on folders `src` and `test`, modify them accordingly
for your cluster architecture. Following the command line instructions below you can have both, CPU
and GPU, versions of HEFFTE. You may also obtain only one of them.

## Install the GPU enabled HEFFTE

~~~
cd heffte/src
src/
    make -j fft=CUFFT
    make install
~~~

Lines above will produce library `libheffte_gpu.a`.

## Install HEFFTE without GPU functionality

~~~
cd heffte/src
src/
    make -j fft=FFTW3
    make install
~~~

Lines above will produce library `libheffte.a`.

* * *

Running tests
=============

If the installation was performed using CMAKE, then `heffte/build/test/` will contain two
executables for C2C FFTs, `test3d_cpu` and `test3d_gpu`, and two for R2C FFTs, `test3d_cpu_r2c` and
`test3d_gpu_r2c`. If you prefer to get this executables via standard Makefile, then do as follows:

~~~
cd heffte/test
test/
    make -j fft=FFTW3 tests_cpu
    make -j fft=CUFFT tests_gpu
~~~

To run these tests on an MPI supported cluster, follow the examples:

~~~
mpirun -n12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c point
mpirun -n12 ./test3d_cpu_r2c -g 512 256 512 -v -i 82783 -c all -pin 1 3 4 -pout 3 4 1
mpirun -n12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c point -pin 1 3 4
mpirun -n12 ./test3d_cpu -g 512 256 512 -v -i 82783 -c all -pout 3 4 1
mpirun -n2  ./test3d_gpu -g 512 256 512 -v -i 82783 -c point -verb
mpirun -n1  ./test3d_gpu_r2c -g 512 256 512 -v -i 82783 -c all
~~~

To run on Summit supercomputer,follow the examples:

~~~
jsrun --smpiargs="-gpu" -n192 -a1 -c1 -g1 -r6 ./test3d_gpu -g 1024 1024 1024 -i 82783 -v -c point
jsrun  -n2560 -a1 -c1 -r40 ./test3d_cpu -g 1024 1024 1024 -i 82783 -v -c all
~~~

* * *

Documentation
=============

To access a detailed documentation on HEFFTE classes and functions you can compile a Doxygen
document by simply typing `make` within this folder.
