#include "test_common.h"

#ifndef TEST_UNITS_STOCK_COMPLEX_H
#define TEST_UNITS_STOCK_COMPLEX_H

template<typename F, size_t L, typename Op1, typename Op2>
void test_stock_complex_avx_template(std::array<std::complex<F>,L> ref1, std::array<std::complex<F>,L> ref2, Op1 stdOp, Op2 stockOp) {
    constexpr size_t L2 = L*2;
    heffte::stock::Complex<F,L2> comp1 {ref1.data()};
    heffte::stock::Complex<F,L2> comp2 {ref2.data()};
    heffte::stock::Complex<F,L2> comp3 = stockOp(comp1, comp2);
    std::vector<std::complex<F>> refVec {};
    std::vector<std::complex<F>> compVec {};
    for(int i = 0; i < L; i++) {
        refVec.push_back(stdOp(ref1[i], ref2[i]));
        compVec.push_back(comp3[i]);
    }
    sassert(approx(refVec, compVec));
}
template<typename F, size_t L, typename Op1, typename Op2>
void test_stock_complex_avx_template(std::array<std::complex<F>,L> ref, F scalar, Op1 stdOp, Op2 stockOp) {
    constexpr size_t L2 = L*2;
    heffte::stock::Complex<F,L2> comp_in {ref.data()};
    heffte::stock::Complex<F,L2> comp_out = stockOp(comp_in, scalar);
    std::vector<std::complex<F>> refVec {};
    std::vector<std::complex<F>> compVec {};
    for(int i = 0; i < L; i++) {
        refVec.push_back(stdOp(ref[i], scalar));
        compVec.push_back(comp_out[i]);
    }
    sassert(approx(refVec, compVec));
}

/*!
 *  \brief This is to test the non-AVX stock::Complex numbers
 *  This must be separated from the AVX case because it's impossible
 *  to disambiguate testing stock::Complex<double,2> from testing
 *  stock::Complex<double,1> using the above method.
 */
template<typename F, typename Op1, typename Op2>
void test_stock_complex_template(std::array<std::complex<F>,1> ref1, std::array<std::complex<F>,1> ref2, Op1 stdOp, Op2 stockOp) {
    heffte::stock::Complex<F,1> comp1 {ref1.data()};
    heffte::stock::Complex<F,1> comp2 {ref2.data()};
    heffte::stock::Complex<F,1> comp3 = stockOp(comp1, comp2);
    std::vector<std::complex<F>> refVec {stdOp(ref1[0], ref2[0])};
    std::vector<std::complex<F>> compVec {comp3[0]};
    sassert(approx(refVec, compVec));
}
//! \brief See test_stock_complex_template
template<typename F, typename Op1, typename Op2>
void test_stock_complex_template(std::array<std::complex<F>,1> ref, F scalar, Op1 stdOp, Op2 stockOp) {
    heffte::stock::Complex<F,1> comp_in {ref.data()};
    heffte::stock::Complex<F,1> comp_out = stockOp(comp_in, scalar);
    std::vector<std::complex<F>> refVec {stdOp(ref[0], scalar)};
    std::vector<std::complex<F>> compVec {comp_out[0]};
    sassert(approx(refVec, compVec));
}

template<typename F, size_t L>
void test_stock_complex_avx_type() {
    std::string test_name = "stock complex<";
    test_name += (std::is_same<F,float>::value ? "float" : "double" );
    test_name += ", " + std::to_string(L) + ">";

    current_test<F, using_nompi> name(test_name);
    using Complex = heffte::stock::Complex<F,L>;
    using stdcomp = std::complex<F>;
    auto std_add = [](stdcomp x, stdcomp y) { return x + y; };
    auto std_sub = [](stdcomp x, stdcomp y) { return x - y; };
    auto std_mul = [](stdcomp x, stdcomp y) { return x * y; };
    auto std_div = [](stdcomp x, stdcomp y) { return x / y; };
    auto std_add_f = [](stdcomp x, F y) { return x + y; };
    auto std_sub_f = [](stdcomp x, F y) { return x - y; };
    auto std_mul_f = [](stdcomp x, F y) { return x * y; };
    auto std_div_f = [](stdcomp x, F y) { return x / y; };

    auto stock_add = [](Complex x, Complex y) { return x + y; };
    auto stock_sub = [](Complex x, Complex y) { return x - y; };
    auto stock_mul = [](Complex x, Complex y) { return x * y; };
    auto stock_div = [](Complex x, Complex y) { return x / y; };

    auto stock_add_f = [](Complex x, F y) { return x + y; };
    auto stock_sub_f = [](Complex x, F y) { return x - y; };
    auto stock_mul_f = [](Complex x, F y) { return x * y; };
    auto stock_div_f = [](Complex x, F y) { return x / y; };

    constexpr size_t L2 = (size_t) std::ceil(((float) L)/2.);
    std::array<std::complex<F>,L2> ref1 {};
    std::array<std::complex<F>,L2> ref2 {};
    for(size_t i = 0; i < L2; i++) {
        F i2 = (F) 2*i;
        ref1[i] = std::complex<F> {i2, i2 + 1.f};
        ref2[i] = std::complex<F> {L-(i2+1.f), L-(i2+2.f)};
    }

    test_stock_complex_avx_template(ref1, ref2, std_add, stock_add);
    test_stock_complex_avx_template(ref1, ref2, std_sub, stock_sub);
    test_stock_complex_avx_template(ref1, ref2, std_mul, stock_mul);
    test_stock_complex_avx_template(ref1, ref2, std_div, stock_div);

    F scalar = 5.;
    test_stock_complex_avx_template(ref1, scalar, std_add_f, stock_add_f);
    test_stock_complex_avx_template(ref1, scalar, std_sub_f, stock_sub_f);
    test_stock_complex_avx_template(ref1, scalar, std_mul_f, stock_mul_f);
    test_stock_complex_avx_template(ref1, scalar, std_div_f, stock_div_f);
}

template<typename F>
void test_stock_complex_type() {
    std::string test_name = "stock complex<";
    test_name += (std::is_same<F,float>::value ? "float" : "double" );
    test_name +=  ", 1>";

    current_test<F, using_nompi> name(test_name);
    using Complex = heffte::stock::Complex<F,1>;
    using stdcomp = std::complex<F>;
    auto std_add = [](stdcomp x, stdcomp y) { return x + y; };
    auto std_sub = [](stdcomp x, stdcomp y) { return x - y; };
    auto std_mul = [](stdcomp x, stdcomp y) { return x * y; };
    auto std_div = [](stdcomp x, stdcomp y) { return x / y; };
    auto std_add_f = [](stdcomp x, F y) { return x + y; };
    auto std_sub_f = [](stdcomp x, F y) { return x - y; };
    auto std_mul_f = [](stdcomp x, F y) { return x * y; };
    auto std_div_f = [](stdcomp x, F y) { return x / y; };

    auto stock_add = [](Complex x, Complex y) { return x + y; };
    auto stock_sub = [](Complex x, Complex y) { return x - y; };
    auto stock_mul = [](Complex x, Complex y) { return x * y; };
    auto stock_div = [](Complex x, Complex y) { return x / y; };

    auto stock_add_f = [](Complex x, F y) { return x + y; };
    auto stock_sub_f = [](Complex x, F y) { return x - y; };
    auto stock_mul_f = [](Complex x, F y) { return x * y; };
    auto stock_div_f = [](Complex x, F y) { return x / y; };

    std::array<std::complex<F>,1> ref1 {std::complex<F>((F) 1., (F) 2.)};
    std::array<std::complex<F>,1> ref2 {std::complex<F>((F) 4., (F) 3.)};

    test_stock_complex_template(ref1, ref2, std_add, stock_add);
    test_stock_complex_template(ref1, ref2, std_sub, stock_sub);
    test_stock_complex_template(ref1, ref2, std_mul, stock_mul);
    test_stock_complex_template(ref1, ref2, std_div, stock_div);

    F scalar = 5.;
    test_stock_complex_template(ref1, scalar, std_add_f, stock_add_f);
    test_stock_complex_template(ref1, scalar, std_sub_f, stock_sub_f);
    test_stock_complex_template(ref1, scalar, std_mul_f, stock_mul_f);
    test_stock_complex_template(ref1, scalar, std_div_f, stock_div_f);
}

#endif // End TEST_UNITS_STOCK_COMPLEX_H