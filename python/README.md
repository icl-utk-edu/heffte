HEFFTE: Python Interface
=======================

HeFFTe Python API has been built in such a way that it mimics our [C++ interface](https://mkstoyanov.bitbucket.io/heffte/md_doxygen_basic_usage.html). This API has been tested with Python 3.6, and can be imported as follows:

~~~
from heffte import *
~~~

The main two kernels are those for plan definition and execution:

~~~
# create plan
fft = heffte.fft3d(heffte.backend.fftw, inboxes[me], outboxes[me], mpi_comm)

# Forward FFT computation
fft.forward(work, work2, heffte.scale.none)
~~~

The definition of inboxes and outboxes are normally provided by the user. However, functions to do everything from scratch are available, c.f., `speed3d.py`.

## Set up for experimentation

The first step is to [install heFFTe](https://mkstoyanov.bitbucket.io/heffte/md_doxygen_installation.html) with the desired backend options. 

Next, set up the environment adding heFFTe library path:

~~~
export heffte_lib_path=/heffte/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$heffte_lib_path
export PYTHONPATH=$PYTHONPATH:$heffte_lib_path
~~~

Then, export paths to the corresponding 1DFFT backend libraries:

~~~
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$fftw_lib_path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$mkl_lib_path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$cufft_lib_path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$rocfft_lib_path
~~~


## Performance tests

We provide performance benchmarks on folder `heffte/python`, they can be used for scalability and comparison tests.

~~~
mpirun -n 12 python ./speed3d_r2c fftw double 128 256 256 -p2p -pencils -no-reorder
mpirun -n 5 python --map-by node  ./speed3d_c2c mkl float 1024 256 512  -a2a -slabs -reorder
mpirun -n 2 python ./speed3d_c2c cufft double 512 256 512  -mps -a2a
~~~

For systems, such as Summit supercomputer, which support execution with `jsrun` by default, follow the examples:

~~~
jsrun  -n1280 -a1 -c1 -r40 ./speed3d_r2c fftw double 256 256 256 -pencils 
jsrun --smpiargs="-gpu" -n192 -a1 -c1 -g1 -r6 ./speed3d_c2c cufft double 1024 1024 1024 -p2p -reorder
~~~

Should you have questions about the use of parameter options, please refer to the [flags documentation](https://bitbucket.org/icl/heffte/src/master/doxygen/flags.md) for detailed information. 