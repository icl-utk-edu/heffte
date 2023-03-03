#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh
module purge

export HOME=`pwd`
git clone https://github.com/spack/spack ../spack || true
source ../spack/share/spack/setup-env.sh

VARIANTS=""
[ "$BACKEND" = "FFTW"       ] && VARIANTS="+fftw"
[ "$BACKEND" = "MKL"        ] && VARIANTS="+mkl"
[ "$BACKEND" = "gpu_nvidia" ] && VARIANTS="+cuda cuda_arch=70"
[ "$BACKEND" = "gpu_amd"    ] && VARIANTS="+rocm amdgpu_target=gfx90a"

[ "$STAGE" = "test" ] && RUNTEST="--test=root"

if [ "$STAGE" != "smoketest" ]; then
   spack uninstall -a -y heffte || true
fi
spack dev-build -q --fresh $RUNTEST heffte@master $VARIANTS
if [ "$STAGE" = "smoketest" ]; then
   spack load --first openmpi
   spack test run heffte
fi
