/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_UTILS_H
#define HEFFTE_UTILS_H

#include <algorithm>
#include <stdio.h>

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

#endif /* HEFFTE_UTILS_H */
