/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

void test_factorize(){
    current_test<int, using_nompi> name("prime factorize");

    std::vector<std::array<int, 2>> reference = {{1, 935}, {5, 187}, {11, 85}, {17, 55}, {55, 17}, {85, 11}, {187, 5}};

    auto factors = heffte::get_factors(935);

    sassert(match(factors, reference));

    reference = {{1, 27}, {3, 9}, {9, 3}};
    factors = heffte::get_factors(reference.front()[1]);
    sassert(match(factors, reference));
}

void test_process_grid(){
    current_test<int, using_nompi> name("process grid");

    std::array<int, 2> reference = {4, 5};
    std::array<int, 2> result = heffte::make_procgrid(20);
    sassert(reference == result);

    reference = {1, 17};
    result = heffte::make_procgrid(17);
    sassert(reference == result);

    reference = {81, 81};
    result = heffte::make_procgrid(6561);
    sassert(reference == result);

    reference = {17, 19};
    result = heffte::make_procgrid(323);
    sassert(reference == result);

    reference = {8, 16};
    result = heffte::make_procgrid(128);
    sassert(reference == result);
}

void test_split_pencils(){
    using namespace heffte;
    current_test<int, using_nompi> name("split pencils");

    box3d const world = {{0, 0, 0}, {1, 3, 5}};
    std::vector<box3d> reference = {{{0, 0, 0}, {0, 1, 5}}, {{0, 2, 0}, {0, 3, 5}},
                                    {{1, 0, 0}, {1, 1, 5}}, {{1, 2, 0}, {1, 3, 5}}};
    // note that the order of the boxes moves fastest in the mid-dimension
    // this tests the reordering
    std::vector<box3d> result = make_pencils(world, {2, 2}, 2, reference);
    sassert(match(result, reference));

    std::vector<box3d> reference2 = {{{0, 0, 0}, {1, 1, 2}}, {{0, 2, 0}, {1, 3, 2}},
                                     {{0, 0, 3}, {1, 1, 5}}, {{0, 2, 3}, {1, 3, 5}}};
    std::vector<box3d> result2 = make_pencils(world, {2, 2}, 0, reference);
    sassert(match(result2, reference2));

    box3d const reconstructed_world = find_world({result, result});
    sassert(reconstructed_world == world);
}

template<typename scalar_type>
std::vector<scalar_type> make_input(){
    std::vector<scalar_type> result(24);
    for(int i=0; i<24; i++) result[i] = static_cast<scalar_type>(i + 1);
    return result;
}

template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft0(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<result.size(); i+=2){
        result[i]   = static_cast<typename fft_output<scalar_type>::type>(3 + 2 * i);
        result[i+1] = static_cast<typename fft_output<scalar_type>::type>(-1.0);
    }
    return result;
}
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft1(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(int j=0; j<4; j++){
        for(int i=0; i<2; i++){
            result[6 * j + i]     = typename fft_output<scalar_type>::type((2*j + i+1) * 9.0 - i * 6.0);
            result[6 * j + i + 2] = typename fft_output<scalar_type>::type(-3.0,  1.73205080756888);
            result[6 * j + i + 4] = typename fft_output<scalar_type>::type(-3.0, -1.73205080756888);
        }
    }
    return result;
}
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft2(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<6; i++){
        result[i]    = typename fft_output<scalar_type>::type(40.0 + 4 * i);
        result[i+ 6] = typename fft_output<scalar_type>::type(-12.0, 12.0);
        result[i+12] = typename fft_output<scalar_type>::type(-12.0);
        result[i+18] = typename fft_output<scalar_type>::type(-12.0, -12.0);
    }
    return result;
}

#ifdef Heffte_ENABLE_FFTW
template<typename scalar_type>
void test_fftw_1d_complex(){
    current_test<scalar_type, using_nompi> name("fftw3 one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::fftw_executor fft(box, i);

        std::vector<scalar_type> result = input;
        fft.forward(result.data());
        sassert(approx(result, reference[i]));

        fft.backward(result.data());
        for(auto &r : result) r /= (2.0 + i);
        sassert(approx(result, input));
    }
}

template<typename scalar_type>
void test_fftw_1d_real(){
    current_test<scalar_type, using_nompi> name("fftw3 one-dimension");

    // make a box
    box3d const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };

    for(size_t i=0; i<reference.size(); i++){
        heffte::fftw_executor fft(box, i);

        std::vector<typename fft_output<scalar_type>::type> result(input.size());
        fft.forward(input.data(), result.data());
        sassert(approx(result, reference[i]));

        std::vector<scalar_type> back_result(result.size());
        fft.backward(result.data(), back_result.data());
        for(auto &r : back_result) r /= (2.0 + i);
        sassert(approx(back_result, input));
    }
}

// tests for the 1D fft
void test_fftw(){
    test_fftw_1d_real<float>();
    test_fftw_1d_real<double>();
    test_fftw_1d_complex<std::complex<float>>();
    test_fftw_1d_complex<std::complex<double>>();
}
#else
void test_fftw(){}
#endif

int main(int argc, char *argv[]){

    all_tests<using_nompi> name("Non-MPI Tests");

    test_factorize();
    test_process_grid();
    test_split_pencils();

    test_fftw();

    return 0;
}
