# Installation

[TOC]

### Requirements

At the minimum, HeFFTe requires a C++11 capable compiler,
an implementation of the Message Passing Library (MPI),
and at least one backend FFT library.
The HeFFTe library can be build with either CMake 3.10 or newer,
or a simple GNU Make build engine.
CMake is the recommended way to use HeFFTe since dependencies and options
are much easier to export to user projects.

| Compiler | Tested versions |
|----|----|
| gcc      | 6 - 8           |
| clang    | 4 - 5           |
| icc      | 18              |
| OpenMPI  | 2.1.1           |

Tested backend libraries:

| Backend    | Tested versions |
|----|----|
| fftw3      | 3.3.7           |
| mkl        | 2016            |
| cuda/cufft | 9.0 - 10.2      |
| rocm/rocfft| 3.7             |

The listed tested versions are part of the continuous integration and nightly build systems,
but HeFFTe may yet work with other compilers and backend versions.

### CMake Installation

Typical CMake build follows the steps:
```
mkdir build
cd build
cmake <cmake-build-command-options> <path-to-heffte-source>
make
make install
make test_install
```

Typical CMake build command:
```
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON     \
    -D CMAKE_INSTALL_PREFIX=<path-for-installation> \
    -D Heffte_ENABLE_FFTW=ON \
    -D FFTW_ROOT=<path-to-fftw3-installation> \
    -D Heffte_ENABLE_CUDA=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=<path-to-cuda-installation> \
    <path-to-heffte-source-code>
```

The standard CMake options are also accepted:
```
    CMAKE_CXX_COMPILER=<path-to-suitable-cxx-compiler>        (sets the C++ compiler)
    CMAKE_CXX_FLAGS="<additional cxx flags>"                  (adds flags to the build process)
    MPI_CXX_COMPILER=<path-to-suitable-mpi-compiler-wrapper>  (specifies the MPI compiler wrapper)
```

Additional HeFFTe options:
```
    Heffte_ENABLE_ROCM=<ON/OFF>      (enable the rocFFT backend)
    Heffte_ENABLE_MKL=<ON/OFF>       (enable the MKL backend)
    MKL_ROOT=<path>                  (path to the MKL folder)
    Heffte_ENABLE_DOXYGEN=<ON/OFF>   (build the documentation)
    Heffte_ENABLE_TRACING=<ON/OFF>   (enable the even logging engine)
```

### List of Available Backend Libraries

* **FFTW3:** the [fftw3](http://www.fftw.org/) library is the de-facto standard for open source FFT library and is distributed under the GNU General Public License, the fftw3 backend is enabled with:
```
    -D Heffte_ENABLE_FFTW=ON
    -D FFTW_ROOT=<path-to-fftw3-installation>
```
Note that fftw3 uses two different libraries for single and double precision, while HeFFTe handles all precisions in a single template library; thus both the fftw3 (double-precision) and fftw3f (single-precision) variants are needed and those can be installed in the same path.

* **MKL:** the [Intel Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) provides optimized FFT implementation targeting Intel processors and can be enabled within HeFFTe with:
```
    -D Heffte_ENABLE_MKL=ON
    -D MKL_ROOT=<path-to-mkl-installation>
```
The `MKL_ROOT` default to the environment variable `MKLROOT` (chosen by Intel). MKL also requires the `iomp5` library, which is the Intel implementation of the OpenMP standard, HeFFTe will find it by default if it is visible in the default CMake search path or the `LD_LIBRARY_PATH`.

* **CUFFT:** the [Nvidia CUDA framework](https://developer.nvidia.com/cuda-zone) provides a GPU accelerated FFT library [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html), which can be enabled in HeFFTe with:
```
    -D Heffte_ENABLE_CUDA=ON
    -D CUDA_TOOLKIT_ROOT_DIR=<path-to-cuda-installation>
```

* **ROCFFT:**  the [AMD ROCm framework](https://github.com/RadeonOpenCompute/ROCm) provides a GPU accelerated FFT library [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT), which can be enabled in HeFFTe with:
```
    -D CMAKE_CXX_COMPILER=hipcc
    -D Heffte_ENABLE_ROCM=ON
```

**Note:** CUDA and ROCM cannot be enabled at the same time and both backends operate with arrays allocated in GPU device memory (or alternatively shared/managed memory). By default when using either GPU backend, HeFFTe assumes that the MPI implementation is CUDA-Aware, see the next section.


### GPU-Aware MPI

Different implementations of MPI can provide GPU-Aware capabilities, where data can be send/received directly in GPU memory. OpenMPI provided CUDA aware capabilities if compiled with the corresponding options, e.g., see [CUDA-Aware OpenMPI](https://www.open-mpi.org/faq/?category=buildcuda). Both CUDA and ROCm support such API; however, the specific implementation available to the user may not be available for various reasons, e.g., insufficient hardware support. HeFFTe can be compiled without GPU-Aware capabilities with the CMake option:
```
    -D Heffte_DISABLE_GPU_AWARE_MPI=ON
```
**Note:** disabling the GPU-Aware capabilities guarantees correctness of the computed results but may have very detrimental impact on performance. The option is provided for testing and debugging and for development of user code on a machine that does not have GPU-Aware support, e.g., an office desktop or a personal laptop.


### Linking to HeFFTe

HeFFTe installs a CMake package-config file in
```
    <install-prefix>/lib/cmake/
```
Typical project linking to HeFFTe will look like this:
```
    project(heffte_user VERSION 1.0 LANGUAGES CXX)

    find_package(Heffte PATHS <install-prefix>)

    add_executable(foo ...)
    target_link_libraries(foo Heffte::Heffte)
```
An example is installed in `<install-prefix>/share/heffte/examples/`.


### GNU Make Installation
HeFFTe supports a GNU Make build engine, where dependencies and compilers
are set manually in the included Makefile.
Selecting the backends is done with:
```
    make backends=fftw,cufft
```
The `backends` should be separated by commas and must have correctly selected
compilers, includes, and libraries. Additional options are available, see
```
    make help
```
and see also the comments inside the Makefile.

Testing is invoked with:
```
    make ctest
```
The library will be build in `./lib/`


### Known Issues

* the current testing suite requires about 3GB of free GPU RAM
    * CUDA seem to reserve 100-200MB of RAM per MPI rank and some tests use 12 ranks
