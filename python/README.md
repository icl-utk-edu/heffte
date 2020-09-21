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

Next, make sure the environment is correctly set up, and contains the path to heFFTe library:

~~~
export heffte_lib_path=<heffte_installation_path>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$heffte_lib_path
export PYTHONPATH=$PYTHONPATH:$heffte_lib_path
~~~


To handle GPU devices we use [Numba library](https://numba.pydata.org/), which can be installed as follows:
~~~
pip install numba
~~~
Or via conda,
~~~
conda install numba
~~~

We currently support NVIDIA and AMD devices, refer to the [installation](https://mkstoyanov.bitbucket.io/heffte/md_doxygen_installation.html)  and [benchmarks](https://mkstoyanov.bitbucket.io/heffte/https://bitbucket.org/icl/heffte/src/master/doxygen/.html) websites for further details.

~~~
from numba import hsa  # For AMD devices
from numba import cuda # For CUDA devices
~~~


## Performance tests

We provide performance benchmarks on folder `heffte/python`, they can be modified or used for scalability and comparison tests.

~~~
mpirun -n 12 python speed3d
mpirun -n 2  python speed3d_gpu
~~~

For systems, such as Summit supercomputer, which support execution with `jsrun` by default, follow the examples:

~~~
jsrun --smpiargs="-gpu" -n24 -a1 -c1 -r3 python speed3d
jsrun --smpiargs="-gpu" -n48 -a1 -c1 -g1 -r6 python speed3d_gpu
~~~