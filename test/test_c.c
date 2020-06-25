/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte.h"
#include "mpi.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef Heffte_ENABLE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

// if the condition fails, call MPI_Abort() and print the file and line
#define hassert(condition) \
    if (! (condition) ){ \
        printf("ERROR: at file: %s  line: %d \n", __FILE__, __LINE__); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \

// macro-template, generates input data
#define make_input(fname, type, complex) \
    type* fname(){ \
        type* x = calloc(32, sizeof(type) * ((complex) ? 2 : 1)); \
        int i; \
        if (complex) \
            for(i=0; i<32; i++) x[2*i] = (type) i; \
        else \
            for(i=0; i<32; i++) x[i] = (type) i; \
        return x; \
    } \

make_input(make_sinput, float, 0)
make_input(make_cinput, float, 1)
make_input(make_dinput, double, 0)
make_input(make_zinput, double, 1)

// macro-template that generates the output on MPI rank 0
#define make_output0(fname, type) \
    type* fname(){ \
        type* x = calloc(64, sizeof(type)); \
        x[ 0] =  992.0; \
        x[ 2] =  -32.0; \
        x[ 3] =   32.0; \
        x[ 4] =  -32.0; \
        x[ 6] =  -32.0; \
        x[ 7] =  -32.0; \
        x[ 8] = -128.0; \
        x[ 9] =  128.0; \
        x[16] = -128.0; \
        x[24] = -128.0; \
        x[25] = -128.0; \
        return x; \
    } \

make_output0(make_coutput0, float)
make_output0(make_zoutput0, double)

// macro-template that generates the output on MPI rank 1
#define make_output1(fname, type) \
    type* fname(){ \
        type* x = calloc(64, sizeof(type)); \
        x[0] = -512.0; \
        return x; \
    } \

make_output1(make_coutput1, float)
make_output1(make_zoutput1, double)

// macro-template, compares two arrays with given number of elements
// returns 1 if any two entries differ by more than the tolerance (0 otherwise)
#define verify(fname, type, num, tolerance) \
    int fname(type const *x, type const *y){ \
        int i; \
        for(i=0; i<num; i++){ \
            if (fabs(x[i] - y[i]) > tolerance){ \
                printf(" Observed error: %f \n", fabs(x[i] - y[i])); \
                return 1; \
            } \
        } \
        return 0; \
    } \

verify(approx_sinput, float, 32, 1.E-4)
verify(approx_cinput, float, 64, 1.E-4)
verify(approx_dinput, double, 32, 1.E-11)
verify(approx_zinput, double, 64, 1.E-11)

verify(approx_coutput, float, 64, 1.E-4)
verify(approx_zoutput, double, 64, 1.E-11)


// perform tests on a CPU backend
void perform_tests(int backend, MPI_Comm const comm){
    int me, ranks;
    MPI_Comm_size(comm, &ranks);
    if (ranks != 2){
        printf("Test must use 2 MPI ranks!\n");
        MPI_Abort(comm, 1);
    }
    MPI_Comm_rank(comm, &me);

    float *sinput = make_sinput();
    float *cinput = make_cinput();
    double *dinput = make_dinput();
    double *zinput = make_zinput();

    float *crefoutput = (me == 0) ? make_coutput0() : make_coutput1();
    double *zrefoutput = (me == 0) ? make_zoutput0() : make_zoutput1();

    float *coutput = calloc(64, sizeof(float));
    double *zoutput = calloc(64, sizeof(double));
    double *workspace = malloc(2 * 96 * sizeof(double));

    float *sresult = NULL;
    double *dresult = NULL;

    int full_low[3] = {0, 0, 0};
    int full_high[3] = {3, 3, 3};
    if (me == 0)
        full_high[2] = 1;
    else
        full_low[2] = 2;

    int r2c_low[3] = {0, 0, 0};
    int r2c_high[3] = {3, 3, 2};
    if (me == 0)
        r2c_high[2] = 1;
    else
        r2c_low[2] = 2;

    heffte_plan plan;
    hassert(heffte_plan_create(backend, full_low, full_high, NULL, full_low, full_high, NULL, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS); // test destorying a plan that's never used

    hassert(heffte_plan_create(backend, full_low, full_high, NULL, full_low, full_high, NULL, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_size_inbox(plan) == 32);
    hassert(heffte_size_outbox(plan) == 32);
    hassert(heffte_size_workspace(plan) == 96);
    hassert(heffte_get_backend(plan) == backend);
    hassert(!heffte_is_r2c(plan));

    // forward and backward variants (single precision)
    heffte_forward_s2c(plan, sinput, coutput, Heffte_SCALE_NONE);
    hassert(approx_coutput(coutput, crefoutput) == Heffte_SUCCESS);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2s(plan, coutput, sresult, Heffte_SCALE_FULL);
    hassert(approx_sinput(sresult, sinput) == Heffte_SUCCESS);
    free(sresult);

    heffte_forward_c2c(plan, cinput, coutput, Heffte_SCALE_SYMMETRIC);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2c(plan, coutput, sresult, Heffte_SCALE_SYMMETRIC);
    hassert(approx_cinput(sresult, cinput) == Heffte_SUCCESS);
    free(sresult);

    heffte_forward_c2c_buffered(plan, cinput, coutput, workspace, Heffte_SCALE_NONE);
    hassert(approx_coutput(coutput, crefoutput) == Heffte_SUCCESS);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2c_buffered(plan, coutput, sresult, workspace, Heffte_SCALE_FULL);
    hassert(approx_cinput(sresult, cinput) == Heffte_SUCCESS);
    free(sresult);

    heffte_forward_s2c_buffered(plan, sinput, coutput, workspace, Heffte_SCALE_FULL);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2s_buffered(plan, coutput, sresult, workspace, Heffte_SCALE_NONE);
    hassert(approx_sinput(sresult, sinput) == Heffte_SUCCESS);
    free(sresult);

    // forward and backward variants (double precision)
    heffte_forward_d2z(plan, dinput, zoutput, Heffte_SCALE_NONE);
    hassert(approx_zoutput(zoutput, zrefoutput) == Heffte_SUCCESS);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2d(plan, zoutput, dresult, Heffte_SCALE_FULL);
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_z2z(plan, zinput, zoutput, Heffte_SCALE_SYMMETRIC);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2z(plan, zoutput, dresult, Heffte_SCALE_SYMMETRIC);
    hassert(approx_zinput(dresult, zinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_z2z_buffered(plan, zinput, zoutput, workspace, Heffte_SCALE_NONE);
    hassert(approx_zoutput(zoutput, zrefoutput) == Heffte_SUCCESS);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2z_buffered(plan, zoutput, dresult, workspace, Heffte_SCALE_FULL);
    hassert(approx_zinput(dresult, zinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_d2z_buffered(plan, dinput, zoutput, workspace, Heffte_SCALE_FULL);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2d_buffered(plan, zoutput, dresult, workspace, Heffte_SCALE_NONE);
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);

    hassert(heffte_plan_create_r2c(backend, full_low, full_high, NULL, r2c_low, r2c_high, NULL, 2, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);

    hassert(heffte_plan_create_r2c(backend, full_low, full_high, NULL, r2c_low, r2c_high, NULL, 2, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_size_inbox(plan) == 32);
    hassert(heffte_size_outbox(plan) == (me == 0) ? 32 : 16);
    hassert(heffte_size_workspace(plan) == (me == 0) ? 96 : 88);
    hassert(heffte_get_backend(plan) == backend);
    hassert(heffte_is_r2c(plan));

    // forward and backward variants (single precision)
    heffte_forward_s2c(plan, sinput, coutput, Heffte_SCALE_NONE);
    hassert(approx_coutput(coutput, crefoutput) == Heffte_SUCCESS);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2s(plan, coutput, sresult, Heffte_SCALE_FULL);
    hassert(approx_sinput(sresult, sinput) == Heffte_SUCCESS);
    free(sresult);

    heffte_forward_s2c_buffered(plan, sinput, coutput, workspace, Heffte_SCALE_FULL);

    sresult = calloc(64, sizeof(float));
    heffte_backward_c2s_buffered(plan, coutput, sresult, workspace, Heffte_SCALE_NONE);
    hassert(approx_sinput(sresult, sinput) == Heffte_SUCCESS);
    free(sresult);

    // forward and backward variants (double precision)
    heffte_forward_d2z(plan, dinput, zoutput, Heffte_SCALE_NONE);
    hassert(approx_zoutput(zoutput, zrefoutput) == Heffte_SUCCESS);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2d(plan, zoutput, dresult, Heffte_SCALE_FULL);
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_d2z_buffered(plan, dinput, zoutput, workspace, Heffte_SCALE_FULL);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2d_buffered(plan, zoutput, dresult, workspace, Heffte_SCALE_NONE);
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);

    free(workspace);
    free(coutput);
    free(zoutput);
    free(crefoutput);
    free(zrefoutput);

    free(sinput);
    free(cinput);
    free(dinput);
    free(zinput);
}

#ifdef Heffte_ENABLE_CUDA
// perform CUDA tests
void perform_tests_cuda(int backend, MPI_Comm const comm){
    // CUDA uses a smaller test, CPU tests cover all code CUDA ensures proper work with GPU arrays
    if (backend != Heffte_BACKEND_CUFFT){
        printf("Test must run with Heffte_BACKEND_CUFFT!\n");
        MPI_Abort(comm, 1);
    }

    int me, ranks;
    MPI_Comm_size(comm, &ranks);
    if (ranks != 2){
        printf("Test must use 2 MPI ranks!\n");
        MPI_Abort(comm, 1);
    }
    MPI_Comm_rank(comm, &me);

    double *dinput = make_dinput();
    double *zinput = make_zinput();

    double *zrefoutput = (me == 0) ? make_zoutput0() : make_zoutput1();

    double *zoutput = calloc(64, sizeof(double));

    double *dresult = NULL;

    double *cuda_dinput;
    hassert( cudaMalloc((void**) &cuda_dinput, 32 * sizeof(double)) == cudaSuccess );
    hassert( cudaMemcpy(cuda_dinput, dinput, 32 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess );
    double *cuda_zinput;
    hassert( cudaMalloc((void**) &cuda_zinput, 64 * sizeof(double)) == cudaSuccess );
    hassert( cudaMemcpy(cuda_zinput, zinput, 64 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess );
    double *cuda_zoutput;
    hassert( cudaMalloc((void**) &cuda_zoutput, 64 * sizeof(double)) == cudaSuccess );
    double *cuda_dresult;
    hassert( cudaMalloc((void**) &cuda_dresult, 64 * sizeof(double)) == cudaSuccess );
    double *cuda_workspace;
    hassert( cudaMalloc((void**) &cuda_workspace, 2 * 96 * sizeof(double)) == cudaSuccess );

    int full_low[3] = {0, 0, 0};
    int full_high[3] = {3, 3, 3};
    if (me == 0)
        full_high[2] = 1;
    else
        full_low[2] = 2;

    int r2c_low[3] = {0, 0, 0};
    int r2c_high[3] = {3, 3, 2};
    if (me == 0)
        r2c_high[2] = 1;
    else
        r2c_low[2] = 2;

    heffte_plan plan;
    hassert(heffte_plan_create(backend, full_low, full_high, NULL, full_low, full_high, NULL, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS); // test destorying a plan that's never used

    hassert(heffte_plan_create(backend, full_low, full_high, NULL, full_low, full_high, NULL, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_size_inbox(plan) == 32);
    hassert(heffte_size_outbox(plan) == 32);
    hassert(heffte_size_workspace(plan) == 96);
    hassert(heffte_get_backend(plan) == backend);
    hassert(!heffte_is_r2c(plan));

    heffte_forward_z2z(plan, cuda_zinput, cuda_zoutput, Heffte_SCALE_NONE);
    hassert( cudaMemcpy(zoutput, cuda_zoutput, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_zoutput(zoutput, zrefoutput) == Heffte_SUCCESS);

    dresult = calloc(64, sizeof(double));
    heffte_backward_z2z(plan, cuda_zoutput, cuda_dresult, Heffte_SCALE_FULL);
    hassert( cudaMemcpy(dresult, cuda_dresult, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_dinput(dresult, zinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_z2z_buffered(plan, cuda_zinput, cuda_zoutput, cuda_workspace, Heffte_SCALE_FULL);

    dresult = calloc(64, sizeof(double));
    hassert( cudaMemcpy(cuda_dresult, dresult, 64 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess );
    heffte_backward_z2z_buffered(plan, cuda_zoutput, cuda_dresult, cuda_workspace, Heffte_SCALE_NONE);
    hassert( cudaMemcpy(dresult, cuda_dresult, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_dinput(dresult, zinput) == Heffte_SUCCESS);
    free(dresult);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);


    hassert(heffte_plan_create_r2c(backend, full_low, full_high, NULL, r2c_low, r2c_high, NULL, 2, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);

    hassert(heffte_plan_create_r2c(backend, full_low, full_high, NULL, r2c_low, r2c_high, NULL, 2, comm, NULL, &plan) == Heffte_SUCCESS);

    hassert(heffte_size_inbox(plan) == 32);
    hassert(heffte_size_outbox(plan) == (me == 0) ? 32 : 16);
    hassert(heffte_size_workspace(plan) == (me == 0) ? 96 : 88);
    hassert(heffte_get_backend(plan) == backend);
    hassert(heffte_is_r2c(plan));

    // forward and backward variants (double precision)
    heffte_forward_d2z(plan, cuda_dinput, cuda_zoutput, Heffte_SCALE_NONE);
    hassert( cudaMemcpy(zoutput, cuda_zoutput, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_zoutput(zoutput, zrefoutput) == Heffte_SUCCESS);

    dresult = calloc(64, sizeof(double));
    hassert( cudaMemcpy(cuda_dresult, dresult, 64 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess );
    heffte_backward_z2d(plan, cuda_zoutput, cuda_dresult, Heffte_SCALE_FULL);
    hassert( cudaMemcpy(dresult, cuda_dresult, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    heffte_forward_d2z_buffered(plan, cuda_dinput, cuda_zoutput, cuda_workspace, Heffte_SCALE_SYMMETRIC);

    dresult = calloc(64, sizeof(double));
    hassert( cudaMemcpy(cuda_dresult, dresult, 64 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess );
    heffte_backward_z2d_buffered(plan, cuda_zoutput, cuda_dresult, cuda_workspace, Heffte_SCALE_SYMMETRIC);
    hassert( cudaMemcpy(dresult, cuda_dresult, 64 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess );
    hassert(approx_dinput(dresult, dinput) == Heffte_SUCCESS);
    free(dresult);

    hassert(heffte_plan_destroy(plan) == Heffte_SUCCESS);

    // clean up the arrays
    hassert( cudaFree(cuda_workspace) == cudaSuccess );
    hassert( cudaFree(cuda_dresult) == cudaSuccess );
    hassert( cudaFree(cuda_zoutput) == cudaSuccess );
    hassert( cudaFree(cuda_dinput) == cudaSuccess );
    hassert( cudaFree(cuda_zinput) == cudaSuccess );

    free(zoutput);
    free(zrefoutput);

    free(dinput);
    free(zinput);
}
#endif

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    if (me == 0){
        printf("\n------------------------------------------------------------------------------\n");
        printf("                       Testing HeFFTe C binding\n");
        printf("------------------------------------------------------------------------------\n\n");
    }

    #ifdef Heffte_ENABLE_FFTW
    perform_tests(Heffte_BACKEND_FFTW, MPI_COMM_WORLD);
    if (me == 0) printf("        Heffte_BACKEND_FFTW        OK\n");
    #endif
    #ifdef Heffte_ENABLE_MKL
    perform_tests(Heffte_BACKEND_MKL, MPI_COMM_WORLD);
    if (me == 0) printf("        Heffte_BACKEND_MKL         OK\n");
    #endif
    #ifdef Heffte_ENABLE_CUDA
    perform_tests_cuda(Heffte_BACKEND_CUFFT, MPI_COMM_WORLD);
    if (me == 0) printf("        Heffte_BACKEND_CUFFT       OK\n");
    #endif

    if (me == 0){
        printf("\n------------------------------------------------------------------------------\n");
        printf("                       HeFFTe C binding: ALL GOOD\n");
        printf("------------------------------------------------------------------------------\n\n");
    }

    MPI_Finalize();

    return 0;
}
