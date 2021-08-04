/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

// This makes it easier to refer to vectors, which are used frequently
template<typename F, int L>
using cvec = heffte::stock::complex_vector<F,L>;

/******************************
 ***** Test Complex Types *****
 *****************************/

template<typename F, size_t L>
void test_stock_complex_type() {
    std::string test_name = "stock complex<";
    test_name += (std::is_same<F,float>::value ? "float" : "double" );
    test_name += ", " + std::to_string(L) + ">";

    current_test<F, using_nompi> name(test_name);
    using Complex = heffte::stock::Complex<F,L>;
    using stdcomp = std::complex<F>;

    constexpr size_t vec_sz = (L == 1) ? L : L/2;
    std::vector<std::complex<F>> inp_left {};
    std::vector<std::complex<F>> inp_right {};
    std::vector<std::complex<F>> inp_last {};
    for(size_t i = 0; i < vec_sz; i++) {
        F i2 = (F) 2*i;
        inp_left.push_back(std::complex<F> {i2, i2 + 1.f});
        inp_right.push_back(std::complex<F> {L-(i2+1.f), L-(i2+2.f)});
        inp_last.push_back(std::complex<F> {2*i2, 2*i2 + 2.f});
    }

    Complex comp_left {inp_left.data()};
    Complex comp_right {inp_right.data()};
    Complex comp_last {inp_last.data()};

    F scalar = 5.;
    constexpr size_t UNARY_OP_COUNT = 5;
    constexpr size_t BINARY_OP_COUNT = 4;
    constexpr size_t TERNARY_OP_COUNT = 2;
    std::array<std::function<stdcomp(stdcomp)>, UNARY_OP_COUNT> stlVecUnOps {
        [](stdcomp x) { return -x;                                },
        [](stdcomp x) { return conj(x);                           },
        [](stdcomp x) { return stdcomp {std::abs(x),std::abs(x)}; },
        [](stdcomp x) { return x*std::complex<F>{0.,  1.};        },
        [](stdcomp x) { return x*std::complex<F>{0., -1.};        }
    };
    std::array<std::function<Complex(Complex)>, UNARY_OP_COUNT> stockVecUnOps {
        [](Complex x) { return -x;                   },
        [](Complex x) { return x.conjugate();        },
        [](Complex x) { return Complex(x.modulus()); },
        [](Complex x) { return x.__mul_i();          },
        [](Complex x) { return x.__mul_neg_i();      }
    };
    std::array<std::function<stdcomp(stdcomp, stdcomp)>, BINARY_OP_COUNT> stlVecBinOps {
        [](stdcomp x, stdcomp y) { return x + y; },
        [](stdcomp x, stdcomp y) { return x - y; },
        [](stdcomp x, stdcomp y) { return x * y; },
        [](stdcomp x, stdcomp y) { return x / y; }
    };
    std::array<std::function<Complex(Complex, Complex)>, BINARY_OP_COUNT> stockVecBinOps {
        [](Complex x, Complex y) { return x + y; },
        [](Complex x, Complex y) { return x - y; },
        [](Complex x, Complex y) { return x * y; },
        [](Complex x, Complex y) { return x / y; }
    };
    std::array<std::function<stdcomp(stdcomp, F)>, BINARY_OP_COUNT> stlScalarBinOps {
        [](stdcomp x, F y) { return x + y; },
        [](stdcomp x, F y) { return x - y; },
        [](stdcomp x, F y) { return x * y; },
        [](stdcomp x, F y) { return x / y; }
    };
    std::array<std::function<Complex(Complex, F)>, BINARY_OP_COUNT> stockScalarBinOps {
        [](Complex x, F y) { return x + y; },
        [](Complex x, F y) { return x - y; },
        [](Complex x, F y) { return x * y; },
        [](Complex x, F y) { return x / y; }
    };
    std::array<std::function<stdcomp(stdcomp, stdcomp, stdcomp)>, TERNARY_OP_COUNT> stlVecTriOps {
        [](stdcomp x, stdcomp y, stdcomp z){ return x*y+z;},
        [](stdcomp x, stdcomp y, stdcomp z){ return x*y-z;}
    };
    std::array<std::function<Complex(Complex, Complex, Complex)>, TERNARY_OP_COUNT> stockVecTriOps {
        [](Complex x, Complex y, Complex z){ return x.fmadd(y, z);},
        [](Complex x, Complex y, Complex z){ return x.fmsub(y, z);}
    };

    std::vector<std::complex<F>> ref_out (vec_sz);
    std::vector<std::complex<F>> comp_out (vec_sz);

    Complex comp_out_i;

    for(size_t i = 0; i < UNARY_OP_COUNT; i++) {
        comp_out_i = stockVecUnOps[i](comp_left);
        for(size_t j = 0; j < vec_sz; j++) {
            ref_out[j] = stlVecUnOps[i](inp_left[j]);
            comp_out[j] = comp_out_i[j];
        }
        sassert(approx(ref_out, comp_out));
    }

    for(size_t i = 0; i < BINARY_OP_COUNT; i++) {
        comp_out_i = stockVecBinOps[i](comp_left, comp_right);
        for(size_t j = 0; j < vec_sz; j++) {
            ref_out[j] = stlVecBinOps[i](inp_left[j], inp_right[j]);
            comp_out[j] = comp_out_i[j];
        }
        sassert(approx(ref_out, comp_out));
    }

    for(size_t i = 0; i < BINARY_OP_COUNT; i++) {
        comp_out_i = stockScalarBinOps[i](comp_left, scalar);
        for(size_t j = 0; j < vec_sz; j++) {
            ref_out[j] = stlScalarBinOps[i](inp_left[j], scalar);
            comp_out[j] = comp_out_i[j];
        }
        sassert(approx(ref_out, comp_out));
    }

    for(size_t i = 0; i < TERNARY_OP_COUNT; i++) {
        comp_out_i = stockVecTriOps[i](comp_left, comp_right, comp_last);
        for(size_t j = 0; j < vec_sz; j++) {
            ref_out[j] = stlVecTriOps[i](inp_left[j], inp_right[j], inp_last[j]);
            comp_out[j] = comp_out_i[j];
        }
        sassert(approx(ref_out, comp_out));
    }

}

void test_stock_complex(){
#ifdef __AVX__
    test_stock_complex_type<float,1>();
    test_stock_complex_type<float,4>();
    test_stock_complex_type<float,8>();
#ifdef __AVX512F__
    test_stock_complex_type<float,16>();
#endif
    test_stock_complex_type<double,1>();
    test_stock_complex_type<double,2>();
    test_stock_complex_type<double,4>();
#ifdef __AVX512F__
    test_stock_complex_type<double,8>();
#endif
#else
    test_stock_complex_type<float,1>();
    test_stock_complex_type<double,1>();
#endif
}

/***********************************
 ***** Test Fourier Transforms *****
 **********************************/

// Helper functions for testing 1D Fourier Transforms
template<typename F, int L>
void vec_to_std_complex(std::vector<std::complex<F>>& out, cvec<F,L>& in) {
    constexpr int L2 = L == 1 ? 1 : L/2;
    for(int i = 0; i < in.size(); i++) {
        for(int j = 0; j < L2; j++) out[i*L2 + j] = in[i][j];
    }
}

template<typename F, int L>
std::vector<std::complex<F>> vec_to_std_complex(cvec<F,L>& in) {
    constexpr int L2 = L == 1 ? 1 : L/2;
    std::vector<std::complex<F>> out (L2*in.size());
    vec_to_std_complex(out, in);
    return out;
}

// Template to test a Fourier Transform
template<typename F, int L>
void test_fft_template(int N,
                       std::function<void(cvec<F,L>&,cvec<F,L>&)> fftForward,
                       std::function<void(cvec<F,L>&,cvec<F,L>&)> fftBackward,
                       std::function<void(cvec<F,L>&,cvec<F,L>&)> refForward) {
    cvec<F,L> input {};
    constexpr int L2 = L == 1 ? 1 : L/2;
    std::vector<std::complex<F>>   stl_input {};

    // Test on an impulse signal
    std::complex<F> tmp {1};
    for(int j = 0; j < L2; j++) stl_input.push_back(tmp);
    input.push_back(heffte::stock::Complex<F,L>{tmp});
    for(int i = 1; i < N; i++) {
        tmp = std::complex<F> {0};
        for(int j = 0; j < L2; j++) stl_input.push_back(tmp);
        input.push_back(heffte::stock::Complex<F,L>{tmp});
    }
    cvec<F,L> output_forward_fft    (input.size());
    cvec<F,L> output_forward_ref    (input.size());
    cvec<F,L> output_backward_fft   (input.size());

    fftForward(input, output_forward_fft);
    std::vector<std::complex<F>> stl_output_forward_fft = vec_to_std_complex(output_forward_fft);
    std::vector<std::complex<F>> stl_output_forward_ref = vec_to_std_complex(output_forward_ref);
    for(int i = 0; i < N; i++) for(int j = 0; j < L2; j++) stl_output_forward_ref[i*L2+j] = std::complex<F> {1.};
    sassert(approx(stl_output_forward_fft, stl_output_forward_ref));
    fftBackward(output_forward_fft, output_backward_fft);
    output_backward_fft[0] /= input.size();
    std::vector<std::complex<F>> stl_output_backward_fft = vec_to_std_complex(output_backward_fft);

    // Test on an actual signal comparing to DFT
    for(int i = 0; i < N; i++) {
        tmp = std::complex<F> {(F) (i + 1.)};
        for(int j = 0; j < L2; j++) stl_input[i*L2 + j] = tmp;
        input[i] = heffte::stock::Complex<F,L> {tmp};
    }

    fftForward(input, output_forward_fft);
    refForward(input, output_forward_ref);
    vec_to_std_complex(stl_output_forward_fft, output_forward_fft);
    vec_to_std_complex(stl_output_forward_ref, output_forward_ref);
    sassert(approx(stl_output_forward_fft, stl_output_forward_ref));
    fftBackward(output_forward_fft, output_backward_fft);
    for(auto &r : output_backward_fft) r /= input.size();
    vec_to_std_complex(stl_output_backward_fft, output_backward_fft);
    sassert(approx(stl_output_backward_fft, stl_input));
}

// Testing the DFT
template<typename F, int L>
void test_stock_dft_template() {
    constexpr int L2 = L == 1 ? 1 : L/2;
    constexpr int INPUT_SZ = 11;

    // Represents the calculated DFT of the input [1,2,3,...,11]
    cvec<F,L> reference (INPUT_SZ);
    reference[0] = heffte::stock::Complex<F,L>(66, 0);

    // Represents the imaginary parts of reference
    std::vector<F> imag;
    if(std::is_same<F, float>::value) {
        imag = std::vector<F> {18.73128,  8.5581665,
                                4.765777, 2.5117664,
                                0.7907804};
    }
    else {
        imag = std::vector<F> {18.731279813890875,
                                8.55816705136493,
                                4.765777128986846,
                                2.5117658384695547,
                                0.790780616972353};
    }

    for(int i = 1; i < (INPUT_SZ+1)/2; i++) {
        reference[i] = heffte::stock::Complex<F,L>(-5.5, imag[i-1]);
        reference[INPUT_SZ - i] = heffte::stock::Complex<F,L>(-5.5, -imag[i-1]);
    }

    std::function<void(cvec<F,L>&,cvec<F,L>&)> dftForward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::DFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward);
    };

    std::function<void(cvec<F,L>&,cvec<F,L>&)> dftBackward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::DFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::backward);
    };

    std::function<void(cvec<F,L>&,cvec<F,L>&)> refForward = [&reference](cvec<F,L>& input, cvec<F,L>& output) {
        output = reference;
    };

    test_fft_template(INPUT_SZ, dftForward, dftBackward, refForward);
}

template<typename F>
void test_stock_dft_typed() {
    current_test<F, using_nompi> name("stock DFT test");
    test_stock_dft_template<F,1>();
#ifdef __AVX__
    test_stock_dft_template<F, 4>();
#endif
#ifdef __AVX512F__
    constexpr bool is_float = std::is_same<F, float>::value;
    test_stock_dft_template<F, is_float? 16 : 8>();
#endif
}

void test_stock_dft() {
    test_stock_dft_typed<float>();
    test_stock_dft_typed<double>();
}

// Test the radix-2 Fourier Transform
template<typename F, int L>
void test_stock_pow2_template() {
    constexpr int INPUT_SZ = 1<<4;
    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftForward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::pow2_FFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward);
    };
    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftBackward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::pow2_FFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::backward);
    };
    std::function<void(cvec<F,L>&,cvec<F,L>&)> refForward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::DFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward);
    };
    test_fft_template(INPUT_SZ, fftForward, fftBackward, refForward);
}

template<typename F>
void test_stock_pow2_typed() {
    current_test<F, using_nompi> name("stock FFT radix-2 test");
    test_stock_pow2_template<F,1>();
#ifdef __AVX__
    test_stock_pow2_template<F, 4>();
#endif
#ifdef __AVX512F__
    constexpr bool is_float = std::is_same<F, float>::value;
    test_stock_pow2_template<F, is_float? 16 : 8>();
#endif
}

void test_stock_fft_pow2() {
    test_stock_pow2_typed<float>();
    test_stock_pow2_typed<double>();
}

// Represents the radix-3 Fourier Transform
template<typename F, int L>
void test_stock_pow3_template() {
    constexpr int INPUT_SZ = 9;

    heffte::stock::Complex<F,L> plus120 (-0.5, -sqrt(3)/2.);
    heffte::stock::Complex<F,L> minus120 (-0.5, sqrt(3)/2.);
    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftForward = [&plus120, &minus120](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::pow3_FFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward, plus120, minus120);
    };
    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftBackward = [&plus120, &minus120](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::pow3_FFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::backward, minus120, plus120);
    };
    std::function<void(cvec<F,L>&,cvec<F,L>&)> refForward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::DFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward);
    };
    test_fft_template(INPUT_SZ, fftForward, fftBackward, refForward);
}

template<typename F>
void test_stock_pow3_typed() {
    current_test<F, using_nompi> name("stock FFT radix-3 test");
    test_stock_pow3_template<F, 1>();
#ifdef __AVX__
    test_stock_pow3_template<F, 4>();
#endif
#ifdef __AVX512F__
    constexpr bool is_float = std::is_same<F, float>::value;
    test_stock_pow3_template<F, is_float? 16 : 8>();
#endif
}

void test_stock_fft_pow3() {
    test_stock_pow3_typed<float>();
    test_stock_pow3_typed<double>();
}

// Represents the radix-p Fourier Transform
template<typename F, int L>
void test_stock_composite_template() {
    using node_ptr = std::unique_ptr<stock::biFuncNode<F,L>[]>;
    constexpr int INPUT_SZ = 12;

    int numNodes = stock::getNumNodes(INPUT_SZ);
    node_ptr root (new stock::biFuncNode<F,L>[numNodes]);
    stock::biFuncNode<F,L>* rootPtr = root.get();
    init_fft_tree(rootPtr, INPUT_SZ);

    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftForward = [&rootPtr](cvec<F,L>& input, cvec<F,L>& output){
        heffte::stock::composite_FFT<F,L>(input.data(), output.data(), 1, 1, rootPtr, heffte::direction::forward);
    };

    std::function<void(cvec<F,L>&,cvec<F,L>&)> fftBackward = [&rootPtr](cvec<F,L>& input, cvec<F,L>& output){
        heffte::stock::composite_FFT<F,L>(input.data(), output.data(), 1, 1, rootPtr, heffte::direction::backward);
    };

    std::function<void(cvec<F,L>&,cvec<F,L>&)> refForward = [](cvec<F,L>& input, cvec<F,L>& output) {
        heffte::stock::DFT_helper<F,L>(input.size(), input.data(), output.data(), 1, 1, heffte::direction::forward);
    };

    test_fft_template(INPUT_SZ, fftForward, fftBackward, refForward);
}

template<typename F>
void test_stock_composite_typed() {
    current_test<F, using_nompi> name("stock FFT composite size test");
    test_stock_composite_template<F,1>();
#ifdef __AVX__
    test_stock_composite_template<F, 4>();
#endif
#ifdef __AVX512F__
    constexpr bool is_float = std::is_same<F, float>::value;
    test_stock_composite_template<F, is_float? 16 : 8>();
#endif
}

void test_stock_fft_composite() {
    test_stock_composite_typed<float>();
    test_stock_composite_typed<double>();
}

// Test all stock components
int main(int argc, char** argv) {
    all_tests<using_nompi> name("Non-MPI Tests for Stock Backend");

    test_stock_complex();
    test_stock_dft();
    test_stock_fft_pow2();
    test_stock_fft_pow3();
    test_stock_fft_composite();

    return 0;
}