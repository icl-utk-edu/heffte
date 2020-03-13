
# the -e options means "quit on the first encountered error"
set -e

mkdir -p heffte_post_install_test
cd heffte_post_install_test

rm -f CMakeCache.txt

@CMAKE_COMMAND@ -DCMAKE_CXX_COMPILER=@CMAKE_CXX_COMPILER@ @CMAKE_INSTALL_PREFIX@/share/heffte/examples

make -j3

exist_any=0

echo ""
if [ -f heffte_example_fftw ]; then
    exist_any=1
    echo "running with 2 mpi ranks  ./heffte_example_fftw"
    @MPIEXEC_EXECUTABLE@ @MPIEXEC_NUMPROC_FLAG@ 2 @MPIEXEC_PREFLAGS@ ./heffte_example_fftw @MPIEXEC_POSTFLAGS@
fi

echo ""
