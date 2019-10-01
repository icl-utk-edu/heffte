/**
 * @file
 * GPU functions of HEFFT
 */
 /*
     -- HEFFTE (version 0.1) --
        Univ. of Tennessee, Knoxville
        @date
 */

// 3d scale ffts GPU library

#include <string.h>
#include <stdio.h>
#include "heffte_utils.h"
#include "heffte_scale.h"

template <class T>
__global__ void scale_ffts_kernel(int n, T *data, T fnorm)
{
    int ind = threadIdx.x + 512 * blockIdx.x;
    if(ind < n){
        data[ind] *= fnorm;
    }
}

template
__global__ void scale_ffts_kernel(int n, double *data, double fnorm);
template
__global__ void scale_ffts_kernel(int n, float *data, float fnorm);


// extern "C" void scale_ffts_gpu(int n, double *data, double fnorm)
template <class T>
void scale_ffts_gpu(int n, T *data, T fnorm)
{
#if defined(FFT_CUFFTW) || defined(FFT_CUFFT_A) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
    int  nthreads = 512;
    int  nTB = fft_ceildiv(n, nthreads);
    dim3 grid(nTB);
    dim3 threads(nthreads);
    cudaDeviceSynchronize();
    scale_ffts_kernel<<<grid, threads>>>(n, data, fnorm);
    magma_check_cuda_error();
    cudaDeviceSynchronize();
#else
    exit(-1);
#endif
}

template
void scale_ffts_gpu(int n, double *data, double fnorm);
template
void scale_ffts_gpu(int n, float *data, float fnorm);
