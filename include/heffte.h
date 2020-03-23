/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_H
#define HEFFTE_H

#include <mpi.h>
#include <stdint.h>
#include <math.h>
#include <string.h>


#include "heffte_fft3d_r2c.h"

#ifdef Heffte_ENABLE_FFTW

#if !defined(FFT_MEMALIGN)
#define FFT_MEMALIGN 64
#endif


// Enumerating options for flags
enum{FORWARD, BACKWARD};
enum{HEFFTE_REAL_DATA, HEFFTE_COMPLEX_DATA};

using namespace HEFFTE;


// Initialisation constants
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
double random_init(int &seed);
void error_all(const char *str);
void error_one(const char *str);

int heffte_init();

template <class T>
void heffte_initialize_host(T *work, int n, int seed, int data_type);

template <class T>
void heffte_allocate(heffte_memory_type_t mem, T **work, int fftsize, int64_t &nbytes);

template <class T>
void heffte_deallocate(heffte_memory_type_t mem, T *ptr);

template <class T>
void heffte_set(FFT3d<T> *, const char *, int);

template <class T>
void *heffte_get(FFT3d<T> *, const char *);

int heffte_init(int nthreads);
void heffte_cleanup();


template <class T>
void heffte_validate(T* work, int n, int seed, double &epsmax, MPI_Comm world);

template <class T>
void heffte_plan_create(FFT3d<T> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                        int permute, int *workspace);

template <class T>
void heffte_plan_r2c_create(FFT3d<T> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                            int *workspace);

template <class T>
void heffte_setup_memory(FFT3d<T> *fft, T *sendbuf, T *recvbuf);

template <class T>
void heffte_execute(FFT3d<T> *fft, T *data_in, T *data_out, int flag);

template <class T>
void heffte_execute_r2c(FFT3d<T> *fft, T *data_in, T *data_out);

template <class T>
void heffte_only_1d_ffts(FFT3d<T> *fft, T *in, int flag);

template <class T>
void heffte_only_reshapes(FFT3d<T> *fft, T *in, T *out, int flag);

template <class T>
void heffte_only_one_reshape(FFT3d<T> *fft, T *in, T *out, int flag, int which);

// Grid processor
void heffte_grid_setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                       int* proc_i, int* proc_o, int me, int &nfft_in, int &nfft_out);



void heffte_proc_setup(int *N, int *proc_grid, int nprocs);
void heffte_proc3d(int *N, int &px, int &py, int &pz, int nprocs);

#endif

#endif     /* HEFFTE_H */
