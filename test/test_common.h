/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "heffte.h"

#define tassert(_result_)          \
    if (!(_result_)){              \
        heffte_test_pass = false;  \
        heffte_all_tests = false;  \
        throw std::runtime_error( std::string("mpi rank = ") + std::to_string(heffte::mpi::comm_rank(MPI_COMM_WORLD)) + "  test " \
                                  + heffte_test_name + " in file: " + __FILE__ + " line: " + std::to_string(__LINE__) );          \
    }

#define sassert(_result_)          \
    if (!(_result_)){              \
        heffte_test_pass = false;  \
        heffte_all_tests = false;  \
        throw std::runtime_error( std::string("  test ") \
                                  + heffte_test_name + " in file: " + __FILE__ + " line: " + std::to_string(__LINE__) );          \
    }

using namespace heffte;

using std::cout;
using std::cerr;
using std::endl;
using std::setw;

std::string heffte_test_name;   // the name of the currently running test
bool heffte_test_pass  = true;  // helps in reporting whether the last test passed
bool heffte_all_tests  = true;  // reports total result of all tests

constexpr int pad_type  = 10;
constexpr int pad_large = 50;
constexpr int pad_pass  = 18;
constexpr int pad_all = pad_type + pad_large + pad_pass + 2;

struct using_mpi{};
struct using_nompi{};

template<typename mpi_tag = using_mpi>
struct all_tests{
    all_tests(char const *cname) : name(cname), separator(pad_all, '-'){
        if (std::is_same<mpi_tag, using_nompi>::value or heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
            int const pad = pad_all / 2 + name.size() / 2;
            cout << "\n" << separator << "\n";
            cout << setw(pad) << name << "\n";
            cout << separator << "\n\n";
        }
    }
    ~all_tests(){
        if (std::is_same<mpi_tag, using_nompi>::value or heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
            int const pad = pad_all / 2 + name.size() / 2 + 3;
            cout << "\n" << separator << "\n";
            cout << setw(pad) << name  + "  " + ((heffte_all_tests) ? "pass" : "fail") << "\n";
            cout << separator << "\n\n";
        }
    }
    std::string name;
    std::string separator;
};

template<typename scalar_variant> std::string get_variant(){ return ""; }
template<> std::string get_variant<float>(){ return "float"; }
template<> std::string get_variant<double>(){ return "double"; }
template<> std::string get_variant<std::complex<float>>(){ return "ccomplex"; }
template<> std::string get_variant<std::complex<double>>(){ return "zcomplex"; }

template<typename scalar_variant = int, typename mpi_tag = using_mpi, typename backend_tag = void>
struct current_test{
    current_test(std::string const &name, MPI_Comm const comm) : test_comm(comm){
        static_assert(std::is_same<mpi_tag, using_mpi>::value, "current_test cannot take a comm when using nompi mode");
        heffte_test_name = name;
        heffte_test_pass = true;
        if (std::is_same<mpi_tag, using_mpi>::value) MPI_Barrier(test_comm);
    };
    current_test(std::string const &name) : test_comm(MPI_COMM_NULL){
        static_assert(std::is_same<mpi_tag, using_nompi>::value, "current_test requires a comm when working in mpi mode");
        heffte_test_name = name;
        heffte_test_pass = true;
    };
    ~current_test(){
        if (std::is_same<mpi_tag, using_nompi>::value or heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
            cout << setw(pad_type)  << get_variant<scalar_variant>();
            if (std::is_same<backend_tag, void>::value){
                cout << setw(pad_large) << heffte_test_name;
            }else{
                 cout << setw(pad_large) << heffte_test_name + "<" + heffte::backend::name<backend_tag>() + ">";
            }
            cout << setw(pad_pass)  << ((heffte_test_pass) ? "pass" : "fail") << endl;
        }
        if (std::is_same<mpi_tag, using_mpi>::value) MPI_Barrier(test_comm);
    };
    MPI_Comm const test_comm;
};

template<typename T>
inline bool match(std::vector<T> const &a, std::vector<T> const &b){
    if (a.size() != b.size()) return false;
    for(size_t i=0; i<a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

template<typename T> struct precision{};
template<> struct precision<float>{ static constexpr float tolerance = 1.E-6; };
template<> struct precision<double>{ static constexpr double tolerance = 1.E-11; };
template<> struct precision<std::complex<float>>{ static constexpr float tolerance = 1.E-6; };
template<> struct precision<std::complex<double>>{ static constexpr double tolerance = 1.E-11; };

template<typename T>
inline bool approx(std::vector<T> const &a, std::vector<T> const &b, double correction = 1.0){
    if (a.size() != b.size()) return false;
    for(size_t i=0; i<a.size(); i++)
        if (std::abs(a[i] - b[i]) * correction > precision<T>::tolerance) return false;
    return true;
}

#ifdef Heffte_ENABLE_CUDA
template<typename T>
inline bool match(heffte::cuda::vector<T> const &a, std::vector<T> const &b){
    return match(cuda::unload(a), b);
}
template<typename T>
inline bool approx(heffte::cuda::vector<T> const &a, std::vector<T> const &b, double correction = 1.0){
    return approx(cuda::unload(a), b, correction);
}
#endif

#endif
