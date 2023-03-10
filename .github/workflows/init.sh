source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

alias print=echo
alias load="spack load --first"

load gcc@8
load cmake
MPI="openmpi"
if [ "$BACKEND" = "gpu_amd" ]; then
   MPI="mpich+rocm"
elif [ "$BACKEND" = "gpu_nvidia" ]
   MPI="mpich+cuda"
fi
load $MPI
