/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <iostream>
#include <iomanip>
#include <string>

#include "heffte.h"

#define tassert(_result_)          \
    if (!(_result_)){              \
        heffte_test_pass = false;  \
        heffte_all_tests = false;  \
        throw std::runtime_error( std::string("mpi rank = ") + std::to_string(heffte::mpi::comm_rank(MPI_COMM_WORLD)) + "  test " \
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

struct all_tests{
    all_tests(char const *cname) : name(cname), separator(pad_all, '-'){
        if (heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
            int const pad = pad_all / 2 + name.size() / 2;
            cout << "\n" << separator << "\n";
            cout << setw(pad) << name << "\n";
            cout << separator << "\n\n";
        }
    }
    ~all_tests(){
        if (heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
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
template<> std::string get_variant<std::complex<float>>(){ return "fcomplex"; }
template<> std::string get_variant<std::complex<double>>(){ return "dcomplex"; }

template<typename scalar_variant = int>
struct current_test{
    current_test(std::string const &name, MPI_Comm const comm) : test_comm(comm){
        heffte_test_name = name;
        heffte_test_pass = true;
        MPI_Barrier(test_comm);
    };
    ~current_test(){
        if (heffte::mpi::comm_rank(MPI_COMM_WORLD) == 0){
            cout << setw(pad_type)  << get_variant<scalar_variant>()
                 << setw(pad_large) << heffte_test_name
                 << setw(pad_pass)  << ((heffte_test_pass) ? "pass" : "fail") << endl;
        }
        MPI_Barrier(test_comm);
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

#endif
