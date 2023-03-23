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

[ "$STAGE" = "test" ] && RUNTEST="--test=root"

if [ "$STAGE" = "smoketest" ]; then
   spack load --first $MPI
   spack test run heffte
else
   spack uninstall -a -y heffte || true
   spack install --only=dependencies --fresh heffte@master $VARIANTS ^$MPI
   spack dev-build -i --fresh $RUNTEST heffte@master $VARIANTS ^$MPI
fi
