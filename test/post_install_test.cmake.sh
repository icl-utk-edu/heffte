
# the -e options means "quit on the first encountered error"
set -e

mkdir -p heffte_post_install_test
cd heffte_post_install_test

rm -f CMakeCache.txt

@CMAKE_COMMAND@ \
    -DCMAKE_CXX_COMPILER=@CMAKE_CXX_COMPILER@ \
    -DHeffte_DIR=@CMAKE_INSTALL_PREFIX@/lib/cmake/Heffte \
    -DMPIEXEC_NUMPROC_FLAG="@MPIEXEC_NUMPROC_FLAG@" \
    -DMPIEXEC_PREFLAGS="@MPIEXEC_PREFLAGS@" \
    -DMPIEXEC_POSTFLAGS="@MPIEXEC_POSTFLAGS@" \
    @CMAKE_INSTALL_PREFIX@/share/heffte/testing/

make -j4
ctest -V
