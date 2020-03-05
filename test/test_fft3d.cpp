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
            test_fft3d_arrays<backend::fftw, float, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::fftw, double, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::fftw, std::complex<float>, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::fftw, std::complex<double>, 9, 9, 9>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_const_dest2<backend::cufft>(comm);
            test_fft3d_arrays<backend::cufft, float, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::cufft, double, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::cufft, std::complex<float>, 9, 9, 9>(comm);
            test_fft3d_arrays<backend::cufft, std::complex<double>, 9, 9, 9>(comm);
            #endif
            break;
        case 6:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_vectors<backend::fftw, float, 11, 11, 22>(comm);
            test_fft3d_vectors<backend::fftw, double, 11, 11, 22>(comm);
            test_fft3d_vectors<backend::fftw, std::complex<float>, 11, 11, 22>(comm);
            test_fft3d_vectors<backend::fftw, std::complex<double>, 11, 11, 22>(comm);
            #endif
            break;
        case 8:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_vectors<backend::fftw, float, 16, 15, 15>(comm);
            test_fft3d_vectors<backend::fftw, double, 16, 15, 15>(comm);
            test_fft3d_vectors<backend::fftw, std::complex<float>, 16, 15, 15>(comm);
            test_fft3d_vectors<backend::fftw, std::complex<double>, 16, 15, 15>(comm);
            #endif
            break;
        case 12:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_arrays<backend::fftw, float, 19, 20, 21>(comm);
            test_fft3d_arrays<backend::fftw, double, 19, 20, 21>(comm);
            test_fft3d_arrays<backend::fftw, std::complex<float>, 19, 15, 25>(comm);
            test_fft3d_arrays<backend::fftw, std::complex<double>, 19, 19, 17>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_arrays<backend::cufft, float, 19, 21, 20>(comm);
            test_fft3d_arrays<backend::cufft, double, 19, 20, 21>(comm);
            test_fft3d_arrays<backend::cufft, std::complex<float>, 19, 14, 25>(comm);
            test_fft3d_arrays<backend::cufft, std::complex<double>, 19, 19, 17>(comm);
            #endif

            break;
        default:
            throw std::runtime_error("No test for the given number of ranks!");
    };
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
