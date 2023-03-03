source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

alias print=echo

module try-load gcc@8
module try-load cmake
module try-load openmpi

