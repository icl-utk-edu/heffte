source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

alias print=echo
alias load="spack load --first"

COMPILER=gcc@11.3.0
load $COMPILER

MPI="openmpi~rsh"
if [ "$BACKEND" = "gpu_amd" ]; then
   MPI+="+rocm"
elif [ "$BACKEND" = "gpu_nvidia" ]; then
   MPI+="+cuda"
fi

