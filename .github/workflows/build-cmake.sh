#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

ARGS="-DCMAKE_INSTALL_PREFIX=install"
if [ "$BACKEND" = "MKL" ]; then
   module try-load intel-mkl
   [ -z "$MKLROOT" ] && echo "Error loading MKL!" && exit 1
elif [ "$BACKEND" = "FFTW" ]; then
   module try-load fftw
   fftw-wisdom
elif [[ "$BACKEND" == "ONEAPI" || "$BACKEND" == "gpu_intel" ]]; then
   module try-load intel-oneapi-mkl
   module try-load intel-oneapi-compilers
   BACKEND="ONEAPI"
   ARGS+=" -D CMAKE_CXX_COMPILER=icpx -D Heffte_ONEMKL_ROOT=$MKLROOT"
   [ -z "$MKLROOT" ] && echo "Error loading OneAPI-MKL!" && exit 1
elif [ "$BACKEND" = "gpu_nvidia" ]; then
   BACKEND="CUDA"
   module try-load cuda
   which nvcc
elif [ "$BACKEND" = "gpu_amd" ]; then
   BACKEND="ROCM"
   export PATH=/opt/rocm/bin:$PATH
   which hipcc
else
   # Use the stock backend with AVX instruction set
   BACKEND=AVX
fi

[ "$STAGE" = "build" ] && rm -rf build install || true
mkdir -p build
cd build

if [ "$STAGE" = "build" ]; then
   cmake $ARGS -DHeffte_ENABLE_$BACKEND=ON ..
   make -j4
   make install
   ls -l install/lib*/libheffte.so
elif [ "$STAGE" = "test" ]; then
   make test
elif [ "$STAGE" = "smoketest" ]; then
   echo Smoke test not implemented
fi

