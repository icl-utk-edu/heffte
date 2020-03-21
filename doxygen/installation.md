# Installation

### Requirements

At the minimum, HeFFTe requires a C++ 2011 capable compiler,
an implementation of the Message Passing Library (MPI)
and at least one backend FFT library.
The HeFFTe library can be build with either CMake 3.10 or newer, or (to-add GNU Make) build engine.

| Compiler | Tested versions |
|----|----|
| gcc      | 6 - 8           |
| clang    | 4 - 5           |
| OpenMPI  | 2.1.1           |

Tested backend libraries:

| Backend    | Tested versions |
|----|----|
| fftw3      | 3.3.7           |
| cuda/cufft | 9.0 - 10.2      |

The listed tested versions are part of the continuous integration and nightly build systems,
but HeFFTe may yet work with other compilers and backend versions.

### CMake Installation

Typical CMake build follows the steps:
```
mkdir build
cd build
cmake <cmake-build-command-options>
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
    Heffte_ENABLE_DOXYGEN=<ON/OFF>   (build the documentation)
```

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

### Known Issues

* at the moment, HeFFTe cannot be build without the FFTW backend
    * work is underway to make FFTW optional
* the GNU Make engine is not supported
    * work is in progress to build using simple GNU Make
* the current testing suite requires about 3GB of free GPU RAM
    * CUDA seem to reserve 100-200MB of RAM per MPI rank and some tests use 12 ranks
    * tested on CUDA 10.2 with just allocating and freeing a vary small array

