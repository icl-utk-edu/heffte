#include "test_common.h"

#ifndef TEST_UNITS_STOCK_COMPLEX_H
#define TEST_UNITS_STOCK_COMPLEX_H

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
    for(size_t i = 0; i < vec_sz; i++) {
        F i2 = (F) 2*i;
        inp_left.push_back(std::complex<F> {i2, i2 + 1.f});
        inp_right.push_back(std::complex<F> {L-(i2+1.f), L-(i2+2.f)});
    }
    F scalar = 5.;
    constexpr size_t Num_Ops = 4;
    std::array<std::function<stdcomp(stdcomp, stdcomp)>, Num_Ops> stlVecOps {
        [](stdcomp x, stdcomp y) { return x + y; },
        [](stdcomp x, stdcomp y) { return x - y; },
        [](stdcomp x, stdcomp y) { return x * y; },
        [](stdcomp x, stdcomp y) { return x / y; }
    };
    std::array<std::function<Complex(Complex, Complex)>, Num_Ops> stockVecOps {
        [](Complex x, Complex y) { return x + y; },
        [](Complex x, Complex y) { return x - y; },
        [](Complex x, Complex y) { return x * y; },
        [](Complex x, Complex y) { return x / y; }
    };
    std::array<std::function<stdcomp(stdcomp, F)>, Num_Ops> stlScalarOps {
        [](stdcomp x, F y) { return x + y; },
        [](stdcomp x, F y) { return x - y; },
        [](stdcomp x, F y) { return x * y; },
        [](stdcomp x, F y) { return x / y; }
    };
    std::array<std::function<Complex(Complex, F)>, Num_Ops> stockScalarOps {
        [](Complex x, F y) { return x + y; },
        [](Complex x, F y) { return x - y; },
        [](Complex x, F y) { return x * y; },
        [](Complex x, F y) { return x / y; }
    };
    Complex comp_left {inp_left.data()};
    Complex comp_right {inp_right.data()};
    std::vector<std::complex<F>> ref_vec_out (vec_sz);
    std::vector<std::complex<F>> comp_vec_out (vec_sz);
    std::vector<std::complex<F>> ref_scalar_out (vec_sz);
    std::vector<std::complex<F>> comp_scalar_out (vec_sz);
    for(size_t i = 0; i < Num_Ops; i++) {
        Complex comp_vec_out_i = stockVecOps[i](comp_left, comp_right);
        Complex comp_scalar_out_i = stockScalarOps[i](comp_left, scalar);
        for(size_t j = 0; j < vec_sz; j++) {
            ref_vec_out[j] = stlVecOps[i](inp_left[j], inp_right[j]);
            ref_scalar_out[j] = stlScalarOps[i](inp_left[j], scalar);
            comp_vec_out[j] = comp_vec_out_i[j];
            comp_scalar_out[j] = comp_scalar_out_i[j];
        }
        sassert(approx(ref_vec_out, comp_vec_out));
        sassert(approx(ref_scalar_out, comp_scalar_out));
    }
}

#endif // End TEST_UNITS_STOCK_COMPLEX_H