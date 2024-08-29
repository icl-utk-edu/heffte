#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

export HOME=`pwd`
git clone https://github.com/spack/spack || true
source spack/share/spack/setup-env.sh
spack config --scope=site add upstreams:i1:install_tree:/apps/spacks/current/opt/spack

VARIANTS=""
if [ "$BACKEND" = "FFTW" ]; then
   VARIANTS="+fftw"
elif [ "$BACKEND" = "MKL" ]; then
   VARIANTS="+mkl"
   # Need to replace deprecated intel-mkl package with oneapi version
   sed -i s/intel-mkl/intel-oneapi-mkl/ spack/var/spack/repos/builtin/packages/heffte/package.py
elif [ "$BACKEND" = "CUDA" ]; then
   VARIANTS="+cuda cuda_arch=70 ^cuda@11.8.0"
elif [ "$BACKEND" = "ROCM" ]; then
   VARIANTS="+rocm amdgpu_target=gfx90a ^hip@5.7.3"
fi

SPEC="heffte@develop $VARIANTS ^openmpi"
echo SPEC=$SPEC

if [ "$STAGE" = "build" ]; then
   spack compiler find
   spack spec $SPEC
   spack dev-build $SPEC
elif [ "$STAGE" = "test" ]; then
   spack uninstall -a -y heffte || true
   spack dev-build -i --test=root $SPEC
else # STAGE = smoketest
   spack test run heffte
fi
