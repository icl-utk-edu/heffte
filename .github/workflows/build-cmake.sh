#!/bin/bash -e

set +x
trap 'echo "# $BASH_COMMAND"' DEBUG

STAGE=$1
BACKEND=$2

if [[ -z "$SPACK_SETUP" || ! -f "$SPACK_SETUP" ]]; then
   echo Error! Environment variable \$SPACK_SETUP must point
   echo to a valid setup-env.sh Spack setup script.
   exit 1
fi
source $SPACK_SETUP

[ "$BACKEND" = "CUDA" ] && ENV=heffte-cuda || ENV=heffte

spack env activate --without-view $ENV

spack load cmake
spack load openmpi

ARGS="-DCMAKE_INSTALL_PREFIX=install"
if [ "$BACKEND" = "MKL" ]; then
   ARGS+=" -DHeffte_ENABLE_MKL=ON"
   spack load intel-oneapi-mkl
   [ -z "$MKLROOT" ] && echo "Error loading MKL!" && exit 1
elif [ "$BACKEND" = "FFTW" ]; then
   ARGS+=" -DHeffte_ENABLE_FFTW=ON"
   spack load fftw
   fftw-wisdom
elif [ "$BACKEND" == "ONEAPI" ]; then
   spack load intel-oneapi-mkl
   spack load intel-oneapi-compilers
   ARGS+=" -DHeffte_ENABLE_ONEAPI=ON"
   ARGS+=" -D CMAKE_CXX_COMPILER=icpx -D Heffte_ONEMKL_ROOT=$MKLROOT"
   [ -z "$MKLROOT" ] && echo "Error loading OneAPI-MKL!" && exit 1
elif [ "$BACKEND" = "CUDA" ]; then
   ARGS+=" -DHeffte_ENABLE_CUDA=ON"
   spack load cuda
   which nvcc
elif [ "$BACKEND" = "ROCM" ]; then
   ARGS+=" -DHeffte_ENABLE_ROCM=ON"
   export PATH=/opt/rocm/bin:$PATH
   which hipcc
else
   # Use the stock backend with AVX instruction set
   ARGS+=" -DHeffte_ENABLE_AVX=ON"
fi

[ "$STAGE" = "build" ] && rm -rf build install || true
mkdir -p build
cd build

if [ "$STAGE" = "build" ]; then
   cmake $ARGS ..
   make -j4
   make install
   ls -lR install/lib*/libheffte.so
elif [ "$BACKEND" = "ONEAPI" ]; then
    echo "Skipping tests due to lack of hardware support."
    exit
elif [ "$BACKEND" = "ROCM" ]; then
    echo "ROCM backend tests require HIP-aware MPI.  Skipping tests."
    exit
elif [ "$STAGE" = "test" ]; then
   ctest -V
elif [ "$STAGE" = "smoketest" ]; then
   make test_install
fi

