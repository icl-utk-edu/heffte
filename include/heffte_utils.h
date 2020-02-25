/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_UTILS_H
#define HEFFTE_UTILS_H

#include <algorithm>
#include <vector>
#include <complex>
#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>
#include <stdio.h>
#include <mpi.h>

// Chosing library for 1D FFTs
#if defined(FFT_MKL) || defined(FFT_MKL_OMP)
  #include "mkl_dfti.h"

#elif defined(FFT_FFTW2)
  #if defined(FFTW_SIZE)
    #include "sfftw.h"
    #include "dfftw.h"
  #else
    #include "fftw.h"
  #endif

#elif defined(FFT_CUFFTW)
  #include "cufftw.h"

#elif defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
  #include <cufft.h>
#else // By default we include FFTW3
  #include "fftw3.h"
#endif

// Timing vector
#define NTIMING_VARIABLES 10
extern double timing_array[NTIMING_VARIABLES];

// ==============================================================================

static int i0=0, i1=1;
static double m1=-1e0, p0=0e0, p1=1e0;

typedef enum {
    PARAM_BLACS_CTX,
    PARAM_RANK,
    PARAM_M,
    PARAM_N,
    PARAM_NB,
    PARAM_SEED,
    PARAM_VALIDATE,
    PARAM_NRHS,
    PARAM_NP,
    PARAM_NQ
} params_enum_t;

void setup_params( int params[], int argc, char* argv[] );

void scalapack_pdplrnt( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed );

void scalapack_pdplghe( double *A,
                        int m, int n,
                        int mb, int nb,
                        int myrow, int mycol,
                        int nprow, int npcol,
                        int mloc,
                        int seed );

// Tools for error handling
#if defined(FFT_CUFFTW) || defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
#include <cuda_runtime_api.h>
#include <cuda.h>

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "rank %d, CUDART: %s = %d (%s) at (%s:%d)\n", keep_rank, #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

#define heffte_check_cuda_error() do { \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(e); \
 } \
} while(0)


///////////////////////////////////////////////////////////////////////////////////////////////////
/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// That is, ceil(x/y)*y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
__host__ __device__
static inline int fft_ceildiv( int x, int y )
{
    return (x + y - 1)/y;
}
__host__ __device__
static inline int fft_roundup( int x, int y )
{
    return fft_ceildiv( x, y ) * y;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
#else
/// For integers x >= 0, y > 0, returns x rounded up to multiple of y.
/// That is, ceil(x/y)*y.
/// For x == 0, this is 0.
/// This implementation does not assume y is a power of 2.
static inline int fft_ceildiv( int x, int y )
{
    return (x + y - 1)/y;
}
static inline int fft_roundup( int x, int y )
{
    return fft_ceildiv( x, y ) * y;
}
#define heffte_check_cuda_error(){}
#define cudaMalloc(x, y){}
#define cudaMallocManaged(x, y){}
#define cudaMallocHost(x, y){}
#define cudaFree(x){}
#define cudaFreeHost(x){}
#define cudaHostRegister(x, y, z){}
#define cudaHostUnregister(x){}
#define cudaMemcpy(x,y,w,z){}

///////////////////////////////////////////////////////////////////////////////////////////////////
#endif

namespace heffte {

using std::cout; // remove when things get more stable
using std::endl; // make sure it is not added to a release

/*!
 * \brief Wrappers to miscellaneous MPI methods giving a more C++-ish interface.
 */
namespace mpi {

/*!
 * \brief Returns the rank of this process within the specified \b comm.
 *
 * \param comm is an MPI communicator associated with the process.
 *
 * \returns the rank of the process within the \b comm
 *
 * Uses MPI_Comm_rank().
 */
inline int comm_rank(MPI_Comm const comm){
    int me;
    MPI_Comm_rank(comm, &me);
    return me;
}
//! \brief Returns \b true if this process has the \b me rank within the MPI_COMM_WORLD (useful for debugging).
inline bool world_rank(int me){ return (comm_rank(MPI_COMM_WORLD) == me); }
//! \brief Returns the rank of this process within the MPI_COMM_WORLD (useful for debugging).
inline int world_rank(){ return comm_rank(MPI_COMM_WORLD); }

//! \brief Write the message and the data from the vector-like \b x, performed only on rank \b me (if positive), otherwise using all ranks.
template<typename vector_like>
void dump(int me, vector_like const &x, std::string const &message){
    if (me < 0 or world_rank(me)){
        cout << message << "\n";
        for(auto i : x) cout << i << "  ";
        cout << endl;
    }
}

/*!
 * \brief Returns the size of the specified communicator.
 *
 * \param comm is an MPI communicator associated with the process.
 *
 * \returns the number of ranks associated with the communicator (i.e., size).
 *
 * Uses MPI_Comm_size().
 */
inline int comm_size(MPI_Comm const comm){
    int nprocs;
    MPI_Comm_size(comm, &nprocs);
    return nprocs;
}
/*!
 * \brief Creates a new sub-communicator from the provided processes in \b comm.
 *
 * \param ranks is a list of ranks associated with the \b comm.
 * \param comm is an active communicator that holds all processes in the \b ranks.
 *
 * \returns a new communicator that uses the selected ranks.
 *
 * Uses MPI_Comm_group(), MPI_Group_incl(), MPI_Comm_create(), MPI_Group_free().
 */
inline MPI_Comm new_comm_form_group(std::vector<int> const &ranks, MPI_Comm const comm){
    MPI_Group orig_group, new_group;
    MPI_Comm_group(comm, &orig_group);
    MPI_Group_incl(orig_group, (int) ranks.size(), ranks.data(), &new_group);
    MPI_Comm result;
    MPI_Comm_create(comm, new_group, &result);
    MPI_Group_free(&orig_group);
    MPI_Group_free(&new_group);
    return result;
}

/*!
 * \brief Calls free on the MPI comm.
 *
 * \param comm is the communicator to be deleted, cannot be used after this call.
 *
 * Uses MPI_Comm_free().
 *
 * Note that the method would use const_cast() to pass the MPI_Comm (which is a pointer)
 * to the delete method. This circumvents the C-style of API that doesn't respect the fact
 * that deleting a const-pointer is an acceptable operation.
 */
inline void comm_free(MPI_Comm const comm){
    if (MPI_Comm_free(const_cast<MPI_Comm*>(&comm)) != MPI_SUCCESS)
        throw std::runtime_error("Could not free a communicator.");
}

/*!
 * \brief Returns the MPI equivalent of the \b scalar C++ type.
 *
 * This template cannot be instantiated which indicated an unknown conversion from C++ to MPI type.
 * \tparam scalar a C++ scalar type, e.g., float, double, std::complex<float>, etc.
 *
 * \returns the MPI equivalent, e.g., MPI_FLOAT, MPI_DOUBLE, MPI_C_COMPLEX, etc.
 */
template<typename scalar> inline MPI_Datatype type_from(){
    // note that "!std::is_same<scalar, scalar>::value" is always false,
    // but will not be checked until the template is instantiated
    static_assert(!std::is_same<scalar, scalar>::value, "The C++ type has unknown MPI equivalent.");
    return MPI_BYTE; // come compilers complain about lack of return statement.
}
//! \brief Specialization to hand the int type.
template<> inline MPI_Datatype type_from<int>(){ return MPI_INT; }
//! \brief Specialization to hand the float type.
template<> inline MPI_Datatype type_from<float>(){ return MPI_FLOAT; }
//! \brief Specialization to hand the double type.
template<> inline MPI_Datatype type_from<double>(){ return MPI_DOUBLE; }
//! \brief Specialization to hand the single-precision complex type.
template<> inline MPI_Datatype type_from<std::complex<float>>(){ return MPI_C_COMPLEX; }
//! \brief Specialization to hand the double-precision complex type.
template<> inline MPI_Datatype type_from<std::complex<double>>(){ return MPI_C_DOUBLE_COMPLEX; }

}

}


#endif /* HEFFTE_UTILS_H */
