
#include <mpi.h>
#include <stdint.h>

#ifdef Heffte_ENABLE_FFTW

extern "C" {

void heffte_create_fortran_d(MPI_Fint, void **);

void heffte_destroy_d(void *);

void heffte_setup_fortran_d(void *, int, int, int,
                         int, int, int, int, int, int,
                         int, int, int, int, int, int,
                         int, int *, int *, int *);

void heffte_compute_d(void *, double *, double *, int);

void alloc_device_d(double *work, int size);
void dealloc_device_d(double *work);



void heffte_create_fortran_s(MPI_Fint, void **);

void heffte_destroy_s(void *);

void heffte_setup_fortran_s(void *, int, int, int,
                         int, int, int, int, int, int,
                         int, int, int, int, int, int,
                         int, int *, int *, int *);

void heffte_compute_s(void *, float *, float *, int);

void alloc_device_s(float *work, int size);
void dealloc_device_s(float *work);
}

#endif
