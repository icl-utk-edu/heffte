#!/bin/bash -e

BACKEND=$1

mydir=$(dirname $0)
source $mydir/init.sh
module purge

if [[ "$BACKEND" == "ONEAPI" || "$BACKEND" == "gpu_intel" ]]; then
   echo "OneAPI backend support not yet in HeFFTe spack package"
   exit
fi

export HOME=`pwd`
git clone https://github.com/spack/spack ../spack || true
source ../spack/share/spack/setup-env.sh

VARIANTS=""
[ "$BACKEND" = "FFTW"       ] && VARIANTS="+fftw"
[ "$BACKEND" = "MKL"        ] && VARIANTS="+mkl"
[ "$BACKEND" = "gpu_nvidia" ] && VARIANTS="+cuda cuda_arch=70"
[ "$BACKEND" = "gpu_amd"    ] && VARIANTS="+rocm amdgpu_target=gfx90a"


spack uninstall -a -y heffte || true
spack dev-build -q --fresh heffte@master $VARIANTS
spack load --first openmpi
spack test run heffte

