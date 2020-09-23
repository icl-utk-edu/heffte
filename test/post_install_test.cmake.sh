
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
    @MPIEXEC_EXECUTABLE@ @MPIEXEC_NUMPROC_FLAG@ 2 @Heffte_mpi_preflags@ ./heffte_example_fftw @Heffte_mpi_postflags@
fi

if [[ "@Heffte_ENABLE_PYTHON@" == "ON" ]]; then
    echo ""
    export PYTHONPATH=@CMAKE_INSTALL_PREFIX@/share/heffte/python:$PYTHONPATH
    echo "import heffte" > hello_world.py
    echo "print('heFFTe python module reports version: {0:s}'.format(heffte.__version__))" >> hello_world.py
    "@PYTHON_EXECUTABLE@" hello_world.py
fi

echo ""
