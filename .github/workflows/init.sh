#source /etc/profile
set +x
set -e
trap 'echo "# $BASH_COMMAND"' DEBUG
#shopt -s expand_aliases

source /apps/spacks/current/github_env/heffte/share/spack/setup-env.sh

