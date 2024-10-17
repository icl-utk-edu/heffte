
# the -e options means "quit on the first encountered error"
set -e

mkdir -p heffte_post_install_test
cd heffte_post_install_test

rm -f CMakeCache.txt

heffte_mpic_compiler=""
heffte_mpif_compiler=""
if [[ ! -z "@MPI_C_COMPILER@" ]]; then
    heffte_mpic_compiler="-DMPI_C_COMPILER=@MPI_C_COMPILER@"
fi
if [[ ! -z "@MPI_Fortran_COMPILER@" ]]; then
    heffte_mpif_compiler="-DMPI_Fortran_COMPILER=@MPI_Fortran_COMPILER@"
fi

@CMAKE_COMMAND@ \
    -DCMAKE_CXX_COMPILER=@CMAKE_CXX_COMPILER@ \
    -DCMAKE_CXX_FLAGS="@CMAKE_CXX_FLAGS@" \
    -DHeffte_DIR=@CMAKE_INSTALL_FULL_LIBDIR@/cmake/Heffte \
    -DMPI_CXX_COMPILER="@MPI_CXX_COMPILER@" \
    $heffte_mpic_compiler \
    $heffte_mpif_compiler \
    -DMPIEXEC_NUMPROC_FLAG="@MPIEXEC_NUMPROC_FLAG@" \
    -DMPIEXEC_PREFLAGS="@MPIEXEC_PREFLAGS@" \
    -DMPIEXEC_POSTFLAGS="@MPIEXEC_POSTFLAGS@" \
    @CMAKE_INSTALL_FULL_DATADIR@/heffte/testing/

make -j4
ctest -V
