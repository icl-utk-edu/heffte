/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

constexpr inline int vec_size(int L) { return (L == 1) ? 1 : L/2; }

inline bool match_verbose(std::string a, std::string b) {
    if (a != b) {
      auto len = std::min(a.size(), b.size());
      std::cout << " a=" << '"' << a << '"' << std::endl;
      std::cout << " b=" << '"' << b << '"' << std::endl;
      std::vector<std::string> a_vec;
      std::vector<std::string> b_vec;
      for (auto const c : a.substr(0, len)) {
        a_vec.emplace_back("'" + std::string{c} + "'");
      }
      for (auto const c : b.substr(0, len)) {
        b_vec.emplace_back("'" + std::string{c} + "'");
      }
      return match_verbose(a_vec, b_vec);
    }
    return true;
}

template<typename F, int L>
std::string stringify(heffte::stock::Complex<F,L> const &value) {
    std::stringstream buf;
    buf << std::fixed << std::setprecision(2);
    buf << value;
    return buf.str();
}

template<typename F, int L>
void test_ostream_template() {
    current_test<F, using_nompi> name("Complex<F,L> stream serialization test");
    {
      // initialize from a repeated value
      heffte::stock::Complex<F,L> value{std::complex<F>{-F{1}, F{2.5}}};
      std::string ref = "";
      for (int n = 0; n < vec_size(L); ++n) {
        ref += ", -1.00 + 2.50i";
      }
      ref = "( " + ref.substr(2u) + " )";
      sassert(match_verbose(stringify(value), ref));
    }
    {
      // initialize from list
      heffte::stock::Complex<F,L> value{
        F{1}, (F)(M_PI), F{2}, -(F)(M_PI), F{3}, (F)(M_PI), F{4}, -(F)(M_PI),
        F{5}, (F)(M_PI), F{6}, -(F)(M_PI), F{7}, (F)(M_PI), F{8}, -(F)(M_PI),
      };
      std::string ref = "";
      for (int n = 0; n < vec_size(L); ++n) {
        ref += ", " + std::to_string(n + 1) + ".00 " + (n % 2 == 0 ? '+' : '-') + " 3.14i";
      }
      ref = "( " + ref.substr(2u) + " )";
      sassert(match_verbose(stringify(value), ref));
    }
}

int main(int, char**) {
    all_tests<using_nompi> name("Non-MPI Tests for Complex Class Helpers");

    test_ostream_template<double, 1>();
    test_ostream_template<float, 1>();
    #ifdef Heffte_ENABLE_AVX
    test_ostream_template<double, 2>();
    test_ostream_template<double, 4>();
    test_ostream_template<float, 4>();
    test_ostream_template<float, 8>();
    #endif
    #ifdef Heffte_ENABLE_AVX512
    test_ostream_template<double, 8>();
    test_ostream_template<float, 16>();
    #endif

    return 0;
}
