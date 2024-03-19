#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

export HOME=`pwd`
git clone https://github.com/spack/spack /tmp/spack
source /tmp/spack/share/spack/setup-env.sh

spack config add upstreams:spack-instance-1:install_tree:/spack/opt/spack

VARIANTS=""
if [ "$BACKEND" = "FFTW" ]; then
   VARIANTS="+fftw"
elif [ "$BACKEND" = "MKL" ]; then
   VARIANTS="+mkl"
   echo HeFFT+mkl in spack uses deprecated intel-mkl package, exiting
   exit
elif [ "$BACKEND" = "CUDA" ]; then
   VARIANTS="+cuda cuda_arch=70 ^cuda@11.8.0"
elif [ "$BACKEND" = "ROCM" ]; then
   VARIANTS="+rocm amdgpu_target=gfx90a ^hip@5.1.3"
fi

SPEC="heffte@develop $VARIANTS ^openmpi~rsh %$COMPILER"
echo SPEC=$SPEC

if [ "$STAGE" = "build" ]; then
   rm -rf .spack
   spack compiler find
   spack spec $SPEC
   spack install --only=dependencies --fresh $SPEC
   spack uninstall -a -y heffte || true
   spack dev-build -i --fresh $SPEC
elif [ "$STAGE" = "test" ]; then
   spack uninstall -a -y heffte || true
   spack dev-build -i --fresh --test=root $SPEC
else # STAGE = smoketest
   spack test run heffte
fi
