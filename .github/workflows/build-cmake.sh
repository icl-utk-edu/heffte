#!/bin/bash -e

BACKEND=$1

mydir=$(dirname $0)
source $mydir/init.sh

ARGS="-DCMAKE_INSTALL_PREFIX=install"
if [ "$BACKEND" == "MKL" ]; then
   module load intel-mkl
   [ -z "$MKLROOT" ] && echo "Error loading MKL!" && exit 1
elif [ "$BACKEND" == "FFTW" ]; then
   module load fftw
   fftw-wisdom
elif [[ "$BACKEND" == "ONEAPI" || "$BACKEND" == "gpu_intel" ]]; then
   module load intel-oneapi-compilers
   BACKEND="ONEAPI"
   ARGS+=" -D CMAKE_CXX_COMPILER=icpx -D Heffte_ONEMKL_ROOT=$MKLROOT"
   [ -z "$MKLROOT" ] && echo "Error loading OneAPI-MKL!" && exit 1
elif [ "$BACKEND" == "gpu_nvidia" ]; then
   BACKEND="CUDA"
   module load cuda
   which nvcc
elif [ "$BACKEND" == "gpu_amd" ]; then
   BACKEND="ROCM"
   export PATH=/opt/rocm/bin:$PATH
   which hipcc
else
   # Use the stock backend with AVX instruction set
   BACKEND=AVX
fi

print "======================================== Setup build"
rm -rf build install
mkdir -p build
cd build

cmake $ARGS -DHeffte_ENABLE_$BACKEND=ON ..

print "======================================== Build"
make -j4

print "======================================== Install"
make install
ls -l install/lib*/libheffte.so

print "======================================== Tests"
make test

