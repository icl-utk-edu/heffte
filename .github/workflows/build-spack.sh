#!/bin/bash -e

STAGE=$1
BACKEND=$2

source $(dirname $0)/init.sh

export HOME=`pwd`
git clone https://github.com/spack/spack ../spack || true
source ../spack/share/spack/setup-env.sh

VARIANTS=""
if [ "$BACKEND" = "FFTW" ];
   VARIANTS="+fftw"
elif [ "$BACKEND" = "MKL" ]
   VARIANTS="+mkl"
elif [ "$BACKEND" = "gpu_nvidia" ]
   VARIANTS="+cuda cuda_arch=70 ^cuda@11.4.3"
elif [ "$BACKEND" = "gpu_amd"    ]
   VARIANTS="+rocm amdgpu_target=gfx90a ^hip@5.1.3"
fi

[ "$STAGE" = "test" ] && RUNTEST="--test=root"

if [ "$STAGE" = "smoketest" ]; then
   spack load --first $MPI
   spack test run heffte
else
   spack uninstall -a -y heffte || true
   spack dev-build -q --fresh $RUNTEST heffte@master $VARIANTS ^$MPI
fi
