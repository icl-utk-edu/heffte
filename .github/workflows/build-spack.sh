#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

export HOME=`pwd`
git clone https://github.com/spack/spack ../spack || true
(
   cd ../spack
   git pull
   source share/spack/setup-env.sh
)

VARIANTS=""
if [ "$BACKEND" = "FFTW" ]; then
   VARIANTS="+fftw"
elif [ "$BACKEND" = "MKL" ]; then
   VARIANTS="+mkl"
elif [ "$BACKEND" = "gpu_nvidia" ]; then
   VARIANTS="+cuda cuda_arch=70 ^cuda@11.4.3"
elif [ "$BACKEND" = "gpu_amd" ]; then
   VARIANTS="+rocm amdgpu_target=gfx90a ^hip@5.1.3"
fi

if [ "$STAGE" = "build" ]; then
   rm -rf .spack
   spack compiler find
   spack uninstall -a -y heffte || true
   spack install --only=dependencies --fresh heffte@master $VARIANTS %$COMPILER ^$MPI
   spack dev-build -i --fresh heffte@master $VARIANTS %$COMPILER ^$MPI
elif [ "$STAGE" = "test" ]; then
   spack dev-build -i --fresh --test=root heffte@master $VARIANTS %$COMPILER ^$MPI
else
   # STAGE = smoketest
   #spack load --first $MPI %$COMPILER
   spack test run heffte
fi
