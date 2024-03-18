source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

alias print=echo

COMPILER=gcc@9.5.0
module load $COMPILER

MPI="openmpi"

