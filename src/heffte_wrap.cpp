/* ----------------------------------------------------------------------
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       heFFTe wrappers
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>

#include "heffte_wrap.h"
#include "heffte_fft3d.h"

using namespace HEFFTE_NS;


/* ----------------------------------------------------------------------
   Create an FFT object
------------------------------------------------------------------------- */

extern "C"
{

void heffte_create_fortran_d(MPI_Fint fcomm, void **ptr)
{
  MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
  FFT3d<double> *fft = new FFT3d<double>(ccomm);
  *ptr = (void *) fft;
}

void heffte_create_fortran_s(MPI_Fint fcomm, void **ptr)
{
  MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
  FFT3d<float> *fft = new FFT3d<float>(ccomm);
  *ptr = (void *) fft;
}


void heffte_compute_d(void *ptr, double *in, double *out, int flag)
{
    FFT3d<double> *fft = (FFT3d<double> *) ptr;
    fft->compute(in,out,flag);
}

void heffte_compute_s(void *ptr, float *in, float *out, int flag)
{
    FFT3d<float> *fft = (FFT3d<float> *) ptr;
    fft->compute(in,out,flag);
}


void heffte_setup_fortran_d(void *ptr,
                         int nfast, int nmid, int nslow,
                         int in_ilo, int in_ihi, int in_jlo,
                         int in_jhi, int in_klo, int in_khi,
                         int out_ilo, int out_ihi, int out_jlo,
                         int out_jhi, int out_klo, int out_khi,
                         int permute, int *fftsize_caller,
                         int *sendsize_caller, int *recvsize_caller)
{
  FFT3d<double> *fft = (FFT3d<double> *) ptr;

  fft->scaled = 1;
  fft->collective = 0;


    #ifdef HEFFTE_GPU
      fft->mem_type = HEFFTE_MEM_GPU;
    #else
      fft->mem_type = HEFFTE_MEM_CPU_ALIGN;
    #endif


  int *N, *i_lo, *i_hi, *o_lo, *o_hi;
  int fftsize, sendsize, recvsize;
  int aux;
  int dimension = 3;
  N = new int[dimension];
  N[0] = nfast; N[1] = nmid; N[2] = nslow;

  i_lo = new int[dimension];
  i_hi = new int[dimension];
  o_lo = new int[dimension];
  o_hi = new int[dimension];
  i_lo[0] = in_ilo-1; i_lo[1] = in_jlo-1; i_lo[2] = in_klo-1;
  i_hi[0] = in_ihi-1; i_hi[1] = in_jhi-1; i_hi[2] = in_khi-1;
  o_lo[0] = out_ilo-1; o_lo[1] = out_jlo-1; o_lo[2] = out_klo-1;
  o_hi[0] = out_ihi-1; o_hi[1] = out_jhi-1; o_hi[2] = out_khi-1;


  fft->setup( N, i_lo, i_hi, o_lo, o_hi, permute, fftsize, sendsize, recvsize);

  *fftsize_caller = fftsize;
  *sendsize_caller = sendsize;
  *recvsize_caller = recvsize;
}


void heffte_setup_fortran_s(void *ptr,
                         int nfast, int nmid, int nslow,
                         int in_ilo, int in_ihi, int in_jlo,
                         int in_jhi, int in_klo, int in_khi,
                         int out_ilo, int out_ihi, int out_jlo,
                         int out_jhi, int out_klo, int out_khi,
                         int permute, int *fftsize_caller,
                         int *sendsize_caller, int *recvsize_caller)
{
  FFT3d<float> *fft = (FFT3d<float> *) ptr;

  fft->scaled = 1;
  fft->collective = 0;


    #ifdef HEFFTE_GPU
      fft->mem_type = HEFFTE_MEM_GPU;
    #else
      fft->mem_type = HEFFTE_MEM_CPU_ALIGN;
    #endif


  int *N, *i_lo, *i_hi, *o_lo, *o_hi;
  int fftsize, sendsize, recvsize;
  int aux;
  int dimension = 3;
  N = new int[dimension];
  N[0] = nfast; N[1] = nmid; N[2] = nslow;

  i_lo = new int[dimension];
  i_hi = new int[dimension];
  o_lo = new int[dimension];
  o_hi = new int[dimension];
  i_lo[0] = in_ilo-1; i_lo[1] = in_jlo-1; i_lo[2] = in_klo-1;
  i_hi[0] = in_ihi-1; i_hi[1] = in_jhi-1; i_hi[2] = in_khi-1;
  o_lo[0] = out_ilo-1; o_lo[1] = out_jlo-1; o_lo[2] = out_klo-1;
  o_hi[0] = out_ihi-1; o_hi[1] = out_jhi-1; o_hi[2] = out_khi-1;

  fft->setup( N, i_lo, i_hi, o_lo, o_hi, permute, fftsize, sendsize, recvsize);

  *fftsize_caller = fftsize;
  *sendsize_caller = sendsize;
  *recvsize_caller = recvsize;
}




void heffte_destroy_d(void *ptr)
{
  FFT3d<double> *fft = (FFT3d<double> *) ptr;
  delete fft;
}

void heffte_destroy_s(void *ptr)
{
  FFT3d<float> *fft = (FFT3d<float> *) ptr;
  delete fft;
}

void alloc_device_d_(double *work, int size)
{
  int64_t  nbytes = ((int64_t) (sizeof(double) * size) );
  cudaMalloc((void**)&work, nbytes);
}

void alloc_device_s_(float *work, int size)
{
  int64_t  nbytes = ((int64_t) (sizeof(float) * size) );
  cudaMalloc((void**)&work, nbytes);
}

void dealloc_device_d_(double *ptr)
{
  cudaFree(ptr);
}
void dealloc_device_s_(float *ptr)
{
  cudaFree(ptr);
}

}
