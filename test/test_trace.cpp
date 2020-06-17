/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_fft3d.h"

void perform_tracing_test(MPI_Comm const comm){
    all_tests<> name("tracing mechanism");
    int const num_ranks = mpi::comm_size(comm);
    assert(num_ranks == 2);

    #if defined(Heffte_ENABLE_FFTW)
    test_fft3d_arrays<backend::fftw, double, 9, 9, 9>(comm);
    #elif defined(Heffte_ENABLE_CUDA)
    test_fft3d_arrays<backend::cufft, double, 9, 9, 9>(comm);
    #endif
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    std::string root_filename = "_tracing_file";
    init_tracing(root_filename);

    perform_tracing_test(MPI_COMM_WORLD);

    finalize_tracing();

    std::string expected_filename = root_filename + "_" + std::to_string(mpi::world_rank()) + ".log";

    std::ifstream test_file(expected_filename);
    tassert(test_file.good());

    MPI_Finalize();

    return 0;
}
