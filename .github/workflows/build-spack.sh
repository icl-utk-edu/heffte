#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

export HOME=`pwd`
if [ ! -d spack ]; then
   git clone -b v0.23.1 https://github.com/spack/spack
fi
source spack/share/spack/setup-env.sh
spack config --scope=site add upstreams:i1:install_tree:/apps/spacks/current/opt/spack

SPEC="heffte@develop "
if [ "$BACKEND" = "FFTW" ]; then
   SPEC+="+fftw"
elif [ "$BACKEND" = "MKL" ]; then
   SPEC+="+mkl"
   # Need to replace deprecated intel-mkl package with oneapi version
   sed -i s/intel-mkl/intel-oneapi-mkl/ spack/var/spack/repos/builtin/packages/heffte/package.py
elif [ "$BACKEND" = "CUDA" ]; then
   SPEC+="+cuda cuda_arch=70 ^cuda@11.8.0"
elif [ "$BACKEND" = "ROCM" ]; then
   SPEC+="+rocm amdgpu_target=gfx90a ^hip@5.7.3"
fi

[ "$BACKEND" = "CUDA" ] && CUDA="+cuda" || CUDA="~cuda"
SPEC+=" ^openmpi~rsh$CUDA ^hwloc$CUDA"
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
