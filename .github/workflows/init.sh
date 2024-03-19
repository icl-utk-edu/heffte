source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
shopt -s expand_aliases

COMPILER=gcc@9.5.0
module load $COMPILER

