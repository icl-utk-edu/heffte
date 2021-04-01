/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_fft3d.h"

template<typename backend_tag>
void test_fft3d_cases(MPI_Comm const comm){
    int const num_ranks = mpi::comm_size(comm);

    switch(num_ranks){
        case 1:
            test_fft3d_arrays<backend_tag, float, 5, 6, 7>(comm);
            test_fft3d_arrays<backend_tag, double, 5, 6, 7>(comm);
            test_fft3d_arrays<backend_tag, std::complex<float>, 5, 6, 7>(comm);
            test_fft3d_arrays<backend_tag, std::complex<double>, 6, 7, 5>(comm);
            break;
        case 2:
            test_fft3d_const_dest2<backend_tag>(comm);
            test_fft3d_arrays<backend_tag, float, 9, 9, 9>(comm);
            test_fft3d_arrays<backend_tag, double, 9, 9, 9>(comm);
            test_fft3d_arrays<backend_tag, std::complex<float>, 9, 9, 9>(comm);
            test_fft3d_arrays<backend_tag, std::complex<double>, 9, 9, 9>(comm);
            break;
        case 4: // rank 4 uses only 2d tests
            test_fft3d_vectors_2d<backend_tag, std::complex<float>, 31, 31>(comm);
            test_fft3d_vectors_2d<backend_tag, std::complex<double>, 10, 10>(comm);
            break;
        case 6:
            test_fft3d_vectors<backend_tag, float, 11, 11, 22>(comm);
            test_fft3d_vectors<backend_tag, double, 11, 11, 22>(comm);
            test_fft3d_vectors<backend_tag, std::complex<float>, 11, 11, 22>(comm);
            test_fft3d_vectors<backend_tag, std::complex<double>, 11, 11, 22>(comm);
            test_fft3d_vectors_2d<backend_tag, std::complex<float>, 11, 11>(comm);
            test_fft3d_vectors_2d<backend_tag, std::complex<double>, 11, 11>(comm);
            break;
        case 8:
            test_fft3d_vectors<backend_tag, float, 16, 15, 15>(comm);
            test_fft3d_vectors<backend_tag, double, 16, 15, 15>(comm);
            test_fft3d_vectors<backend_tag, std::complex<float>, 16, 15, 15>(comm);
            test_fft3d_vectors<backend_tag, std::complex<double>, 16, 15, 15>(comm);
            test_fft3d_vectors<backend_tag, std::complex<float>, 16, 0, 15>(comm); // effectively 2D
            test_fft3d_vectors<backend_tag, std::complex<double>, 16, 0, 15>(comm);
            break;
        case 12:
            test_fft3d_arrays<backend_tag, float, 19, 20, 21>(comm);
            test_fft3d_arrays<backend_tag, double, 19, 20, 21>(comm);
            test_fft3d_arrays<backend_tag, std::complex<float>, 19, 15, 25>(comm);
            test_fft3d_arrays<backend_tag, std::complex<double>, 19, 19, 17>(comm);
            break;
        default:
            throw std::runtime_error("No test for the given number of ranks!");
    }
}

void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft class");

    #ifdef Heffte_ENABLE_FFTW
    test_fft3d_cases<backend::fftw>(comm);
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_fft3d_cases<backend::mkl>(comm);
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_fft3d_cases<backend::cufft>(comm);
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_fft3d_cases<backend::rocfft>(comm);
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_fft3d_cases<backend::onemkl>(comm);
    #endif
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
