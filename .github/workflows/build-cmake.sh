#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

load cmake
load $MPI %$COMPILER

ARGS="-DCMAKE_INSTALL_PREFIX=install"
if [ "$BACKEND" = "MKL" ]; then
   ARGS+=" -DHeffte_ENABLE_MKL=ON"
   load intel-mkl
   [ -z "$MKLROOT" ] && echo "Error loading MKL!" && exit 1
elif [ "$BACKEND" = "FFTW" ]; then
   ARGS+=" -DHeffte_ENABLE_FFTW=ON"
   load fftw
   fftw-wisdom
elif [[ "$BACKEND" == "ONEAPI" || "$BACKEND" == "gpu_intel" ]]; then
   load intel-oneapi-mkl
   load intel-oneapi-compilers
   spack unload $COMPILER
   load gcc@11
   ARGS+=" -DHeffte_ENABLE_ONEAPI=ON"
   ARGS+=" -D CMAKE_CXX_COMPILER=icpx -D Heffte_ONEMKL_ROOT=$MKLROOT"
   [ -z "$MKLROOT" ] && echo "Error loading OneAPI-MKL!" && exit 1
elif [ "$BACKEND" = "gpu_nvidia" ]; then
   ARGS+=" -DHeffte_ENABLE_CUDA=ON"
   load cuda
   which nvcc
elif [ "$BACKEND" = "gpu_amd" ]; then
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
elif [ "$STAGE" = "test" ]; then
   ctest -V
elif [ "$STAGE" = "smoketest" ]; then
   make test_install
fi

