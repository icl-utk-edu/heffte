/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_fft3d.h"

void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft class");
    int const num_ranks = mpi::comm_size(comm);

    switch(num_ranks){
        case 2:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_const_dest2<backend::fftw>(comm);
            test_fft3d_rank2<backend::fftw, float>(comm);
            test_fft3d_rank2<backend::fftw, double>(comm);
            test_fft3d_rank2<backend::fftw, std::complex<float>>(comm);
            test_fft3d_rank2<backend::fftw, std::complex<double>>(comm);
            #endif
            break;
        case 6:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_rank6<backend::fftw, float>(comm);
            test_fft3d_rank6<backend::fftw, double>(comm);
            test_fft3d_rank6<backend::fftw, std::complex<float>>(comm);
            test_fft3d_rank6<backend::fftw, std::complex<double>>(comm);
            #endif
            break;
        default: break;
    };
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
