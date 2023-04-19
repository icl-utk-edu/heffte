/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte.h"

/*
 * Test custom complex type.
 */

template<typename real_type>
struct custom_complex{
    real_type real;
    real_type imag;
};

template<> struct heffte::is_ccomplex<custom_complex<float>> : std::true_type{};
template<> struct heffte::is_zcomplex<custom_complex<double>> : std::true_type{};

/*
 * This can never run, it is supposed to test the types on compile time.
 */
template<typename backend_tag>
void test_types(){
    heffte::box3d<> inbox({0, 0, 0}, {1, 1, 1});
    heffte::box3d<> outbox({0, 0, 0}, {1, 1, 1});
    heffte::fft3d<backend_tag> fft(inbox, outbox, MPI_COMM_NULL);
    std::vector<custom_complex<float>> ccomplex;
    std::vector<custom_complex<double>> dcomplex;
    std::vector<float> floats;
    std::vector<double> doubles;
    fft.forward(ccomplex.data(), ccomplex.data());
    fft.forward(ccomplex.data(), ccomplex.data(), ccomplex.data());
    fft.forward(dcomplex.data(), dcomplex.data());
    fft.forward(dcomplex.data(), dcomplex.data(), dcomplex.data());

    fft.backward(ccomplex.data(), ccomplex.data());
    fft.backward(ccomplex.data(), ccomplex.data(), ccomplex.data());
    fft.backward(dcomplex.data(), dcomplex.data());
    fft.backward(dcomplex.data(), dcomplex.data(), dcomplex.data());

    heffte::fft3d_r2c<backend_tag> fftr(inbox, outbox, 0, MPI_COMM_NULL);
    fftr.forward(floats.data(), ccomplex.data());
    fftr.forward(floats.data(), ccomplex.data(), ccomplex.data());
    fftr.forward(doubles.data(), dcomplex.data());
    fftr.forward(doubles.data(), dcomplex.data(), dcomplex.data());

    fftr.backward(ccomplex.data(), floats.data());
    fftr.backward(ccomplex.data(), floats.data(), ccomplex.data());
    fftr.backward(dcomplex.data(), doubles.data());
    fftr.backward(dcomplex.data(), doubles.data(), dcomplex.data());
}

int main(int, char**){

    std::cerr << " DO NOT RUN THIS CODE \n";
    std::cerr << " THIS IS A COMPILE TEST ONLY \n";

    test_types<heffte::backend::stock>();
    #ifdef Heffte_ENABLE_FFTW
    test_types<heffte::backend::fftw>();
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_types<heffte::backend::mkl>();
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_types<heffte::backend::cufft>();
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_types<heffte::backend::rocfft>();
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_types<heffte::backend::onemkl>();
    #endif

    return 0;
}
