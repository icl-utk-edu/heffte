/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

/*
 * This is a private header that includes the CUDA headers and defines some common macros.
 * The header is not visible through the external API, effectively hiding the CUDA headers
 * to avoid compiler conflicts.
 */

#ifndef FFT_OLD_API_CUDA_H
#define FFT_OLD_API_CUDA_H

#ifdef FFT_CUFFT // FFT_CUFFT is defined for the old interface only
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>


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

#endif



#endif   //  #ifndef FFT_OLD_API_CUDA_H
