/** @class */
/*
    -- heFFTe --
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

struct using_alltoall{};
struct using_pointtopoint{};
template<typename reshape_variant> std::string get_description(){ return ""; }
template<> std::string get_description<using_alltoall>(){ return "heffte::reshape3d_alltoallv"; }
template<> std::string get_description<using_pointtopoint>(){ return "heffte::reshape3d_pointtopoint"; }

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
template<typename T>
inline bool match_verbose(std::vector<T> const &a, std::vector<T> const &b){
    if (a.size() != b.size()) return false;
    for(size_t i=0; i<a.size(); i++){
        if (a[i] != b[i]){
            cout << " mismatch in entry i = " << i << "  with a[i] = " << a[i] << " and b[i] = " << b[i] << endl;
            return false;
        }
    }
    return true;
}

template<typename T> struct precision{};
template<> struct precision<float>{ static constexpr float tolerance = 5.E-4; };
template<> struct precision<double>{ static constexpr double tolerance = 1.E-11; };
template<> struct precision<std::complex<float>>{ static constexpr float tolerance = 5.E-4; };
template<> struct precision<std::complex<double>>{ static constexpr double tolerance = 1.E-11; };

template<typename T>
inline bool approx(std::vector<T> const &a, std::vector<T> const &b, double correction = 1.0){
    if (a.size() != b.size()) return false;
    for(size_t i=0; i<a.size(); i++)
        if (std::abs(a[i] - b[i]) * correction > precision<T>::tolerance){
            cout << " error magnitude: " << std::abs(a[i] - b[i]) * correction << endl;
            return false;
        }
    return true;
}

template<typename backend_tag, typename = void>
struct test_traits{
    template<typename T> using container = typename backend::buffer_traits<backend_tag>::template container<T>;
    template<typename T>
    static container<T> load(std::vector<T> const &x){ return x; }
    template<typename T>
    static std::vector<T> unload(container<T> const &x){ return x; }
};

#ifdef Heffte_ENABLE_CUDA
using gpu_backend = heffte::backend::cufft;
#endif
#ifdef Heffte_ENABLE_ROCM
using gpu_backend = heffte::backend::rocfft;
#endif
#ifdef Heffte_ENABLE_ONEAPI
using gpu_backend = heffte::backend::onemkl;
#endif
#ifdef Heffte_ENABLE_GPU
template<typename T>
inline bool match(heffte::gpu::vector<T> const &a, std::vector<T> const &b){
    return match(heffte::gpu::transfer::unload(a), b);
}
template<typename T>
inline bool approx(heffte::gpu::vector<T> const &a, std::vector<T> const &b, double correction = 1.0){
    return approx(heffte::gpu::transfer().unload(a), b, correction);
}
template<typename T>
inline bool approx(heffte::gpu::vector<T> const &a, heffte::gpu::vector<T> const &b, double correction = 1.0){
    return approx(a, heffte::gpu::transfer::unload(b), correction);
}
template<typename backend_tag>
struct test_traits<backend_tag, typename std::enable_if<backend::uses_gpu<backend_tag>::value, void>::type>{
    template<typename T> using container = gpu::vector<T>;
    template<typename T>
    static container<T> load(std::vector<T> const &x){ return gpu::transfer().load(x); }
    template<typename T>
    static std::vector<T> unload(container<T> const &x){ return gpu::transfer().unload(x); }
};
#endif

//! \brief Converts a set of c-style strings into a single collection (deque) of c++ strings.
inline std::deque<std::string> arguments(int argc, char *argv[]){
    std::deque<std::string> args;
    for(int i=0; i<argc; i++){
        args.push_back(std::string(argv[i]));
    }
    return args;
}

//! \brief Sets the default options for the backend and then modifies those according to the passed arguments.
template<typename backend_tag>
heffte::plan_options args_to_options(std::deque<std::string> const &args){
    heffte::plan_options options = heffte::default_options<backend_tag>();
    for(auto const &s : args){
        if (s == "-reorder"){
            options.use_reorder = true;
        }else if (s == "-no-reorder"){
            options.use_reorder = false;
        }else if (s == "-a2a"){
            options.use_alltoall = true;
        }else if (s == "-p2p"){
            options.use_alltoall = false;
        }else if (s == "-pencils"){
            options.use_pencils = true;
        }else if (s == "-slabs"){
            options.use_pencils = false;
        }
    }
    return options;
}

template<typename backend_tag>
std::vector<heffte::plan_options> make_all_options(){
    std::vector<heffte::plan_options> result;
    for(int shape = 0; shape < 2; shape++){
        for(int reorder = 0; reorder < 2; reorder++){
            for(int alltoall = 0; alltoall < 2; alltoall++){
                heffte::plan_options options = default_options<backend_tag>();
                options.use_pencils = (shape == 0);
                options.use_reorder = (reorder == 0);
                options.use_alltoall = (alltoall == 0);
                result.push_back(options);
            }
        }
    }
    return result;
}

//! \brief If input and output grid of processors are pencils, useful for comparison with other libraries
bool io_pencils(std::deque<std::string> const &args){
    for(auto &s : args)
        if (s == "-io_pencils")
            return true;
    return false;
}

bool has_mps(std::deque<std::string> const &args){
    for(auto &s : args)
        if (s == "-mps")
            return true;
    return false;
}

#endif
