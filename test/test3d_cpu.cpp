/*
 * File: test3d.cpp
 * License: Please see LICENSE file.
 * Testing HEFFTE library on CPUs
 * Created by Alan Ayala on 09/01/2019
 * Email: aayala@icl.utk.edu
 */

#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <typeinfo>

#include "heffte_fft3d.h"
#include "heffte_trace.h"
#include "heffte.h"
#include "heffte_common.h"

using namespace HEFFTE_NS;
double timing_array[NTIMING_VARIABLES];

// memory alignment settings
#if defined(__INTEL_COMPILER)
#ifndef FFT_INTEL_NO_TBB
#define FFT_USE_TBB_ALLOCATOR
#include "tbb/scalable_allocator.h"
#else
#include <malloc.h>
#endif
#endif

int me,nprocs;
MPI_Comm fft_comm;

int N[3];
int proc_i[3], proc_o[3];
int nloop;
int mode;
int permute;
int oflag,tflag,cflag,eflag,pflag,rflag,vflag,verb;
int seed;

int i_lo[3], i_hi[3];             // initial partition of grid
int o_lo[3], o_hi[3];             // final partition of grid
int nfft_in;                      // # of grid pts I own in initial partition
int nfft_out;                     // # of grid pts I own in final partition
int fftsize;                      // FFT buffer size returned by FFT setup
int sendsize,recvsize;            // Buffer size for global reshape and transposition

FFT3d<float> *fft;
// FFT3d <double> *fft;

double timefft,timeinit,timesetup;
double epsmax;

int64_t nbytes;
float *work;
// double *work;

// functions
void read_input(int, char **);
void heffte_output(int, const char *);
void heffte_timing();

// constants
const char *syntax =
  "Syntax: test3d -g Nx Nx Nz -p Px Py Pz -n Nloop -i 82783 \n"
  "               -c point/all/combo -e pencil/brick -p array/ptr/memcpy\n"
  "               -t -r -o -v";

enum{POINT,ALL2ALL,COMBO};
enum{PENCIL,BRICK};
enum{ARRAY,POINTER,MEMCPY};

/* ----------------------------------------------------------------------
   main progam
------------------------------------------------------------------------- */

int main(int narg, char **args)
{
// MPI initialization
  MPI_Init(&narg,&args);
  fft_comm = MPI_COMM_WORLD;
  MPI_Comm_size(fft_comm,&nprocs);
  MPI_Comm_rank(fft_comm,&me);

  heffte_init();

  // fft = new FFT3d<double>(fft_comm);
  fft = new FFT3d<float>(fft_comm);
  read_input(narg,args);

  MPI_Barrier(fft_comm);
  timeinit -= MPI_Wtime();

  heffte_proc_setup(N, proc_i, nprocs);
  heffte_proc_setup(N, proc_o, nprocs);

  heffte_grid_setup(N, i_lo, i_hi, o_lo, o_hi,
                    proc_i, proc_o, me, nfft_in, nfft_out);

  heffte_plan_create(work, fft, N, i_lo, i_hi, o_lo, o_hi,
                     permute, &fftsize, &sendsize, &recvsize); // change to workspace


  heffte_allocate(0, &work, fftsize, nbytes);
  heffte_initialize_host(work, nfft_in, seed);

  MPI_Barrier(fft_comm);
  timeinit += MPI_Wtime();  // End initialization timing

  if (oflag) heffte_output(0,"Initial grid");

// warmup starts
if (mode == 0) {
  for (int i = 0; i < nloop; i++) {
    heffte_execute(fft, work, work, FORWARD);
    heffte_execute(fft, work, work, BACKWARD);
  }
} else if (mode == 1) {
    for (int i = 0; i < nloop; i++) {
      heffte_execute(fft, work, work, FORWARD);
    }
} // warmup ends

  memset(timing_array, 0, NTIMING_VARIABLES*sizeof(double));
  MPI_Barrier(fft_comm);
  trace_init( 2, 0, 0, NULL);

  MPI_Barrier(fft_comm);
  timefft -= MPI_Wtime();
  if (mode == 0) {
    for (int i = 0; i < nloop; i++) {
      heffte_execute(fft, work, work, FORWARD);
      heffte_execute(fft, work, work, BACKWARD);
    }
  } else if (mode == 1) {
      for (int i = 0; i < nloop; i++) {
        heffte_execute(fft, work, work, FORWARD);
      }
  }
  MPI_Barrier(fft_comm);
  timefft += MPI_Wtime();
  timing_array[0] = timefft;

  int nfft;
  if (mode == 0) nfft = 2*nloop;
  else nfft = nloop;

  double nsize = 1.0 * N[0] * N[1] * N[2];
  double logn = log(nsize);
  double nops = 5.0 * nsize * logn *nfft;

  char buf[80];
  snprintf(buf, sizeof(buf), "fft3d.svg");
  trace_finalize( buf, "trace.css", nops );

  if (vflag) heffte_validate(work, nfft_in, seed, epsmax, fft_comm);  // Error validation
  if (oflag) heffte_output(0,"Final grid");

  heffte_timing();
  heffte_deallocate(0,work);

  delete fft;
  MPI_Finalize();
}



void read_input(int narg, char **args)
{
  // defaults
  N[0] = N[1] = N[2] = 8;
  proc_i[0] = proc_i[1] = proc_i[2] = 0;
  proc_o[0] = proc_o[1] = proc_o[2] = 0;
  nloop = 1;
  cflag = COMBO;
  eflag = PENCIL;
  pflag = MEMCPY;
  tflag = 0;
  rflag = 0;
  oflag = 0;
  mode = 0;
  vflag = 0;
  verb  = 0;

// Default initialization
  fft->mem_type = HEFFTE_MEM_CPU;  // setting memory type for ffts
  heffte_set(fft,"collective",cflag);
  heffte_set(fft,"exchange",eflag);
  heffte_set(fft,"pack",pflag);
  heffte_set(fft,"memory",1);
  heffte_set(fft,"scale",1);
  heffte_set(fft,"reshapeonly",rflag);

// parse args
  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(args[iarg],"-h") == 0) {
      error_all(syntax);
    } else if (strcmp(args[iarg],"-g") == 0) {
      if (iarg+4 > narg) error_all(syntax);
      N[0] = atoi(args[iarg+1]);
      N[1] = atoi(args[iarg+2]);
      N[2] = atoi(args[iarg+3]);
      iarg += 4;
    } else if (strcmp(args[iarg],"-pin") == 0) {
      if (iarg+4 > narg) error_all(syntax);
      proc_i[0] = atoi(args[iarg+1]);
      proc_i[1] = atoi(args[iarg+2]);
      proc_i[2] = atoi(args[iarg+3]);
      iarg += 4;
    } else if (strcmp(args[iarg],"-pout") == 0) {
      if (iarg+4 > narg) error_all(syntax);
      proc_o[0] = atoi(args[iarg+1]);
      proc_o[1] = atoi(args[iarg+2]);
      proc_o[2] = atoi(args[iarg+3]);
      iarg += 4;
    } else if (strcmp(args[iarg],"-n") == 0) {
      if (iarg+2 > narg) error_all(syntax);
      nloop = atoi(args[iarg+1]);
      iarg += 2;
    } else if (strcmp(args[iarg],"-i") == 0) {
      if (iarg+2 > narg) error_all(syntax);
        seed = atoi(args[iarg+1]) + me;
      iarg += 2;
    } else if (strcmp(args[iarg],"-c") == 0) {
      if (iarg+2 > narg) error_all(syntax);
      if (strcmp(args[iarg+1],"point") == 0) cflag = POINT;
      else if (strcmp(args[iarg+1],"all") == 0) cflag = ALL2ALL;
      else if (strcmp(args[iarg+1],"combo") == 0) cflag = COMBO;
      else error_all(syntax);
      heffte_set(fft,"collective",cflag);
      iarg += 2;
    } else if (strcmp(args[iarg],"-e") == 0) {
      if (iarg+2 > narg) error_all(syntax);
      if (strcmp(args[iarg+1],"pencil") == 0) eflag = PENCIL;
      else if (strcmp(args[iarg+1],"brick") == 0) eflag = BRICK;
      else error_all(syntax);
      heffte_set(fft,"exchange",eflag);
      iarg += 2;
    } else if (strcmp(args[iarg],"-p") == 0) {
      if (iarg+2 > narg) error_all(syntax);
      if (strcmp(args[iarg+1],"array") == 0) pflag = ARRAY;
      else if (strcmp(args[iarg+1],"ptr") == 0) pflag = POINTER;
      else if (strcmp(args[iarg+1],"memcpy") == 0) pflag = MEMCPY;
      else error_all(syntax);
      heffte_set(fft,"pack",pflag);
      iarg += 2;
    } else if (strcmp(args[iarg],"-t") == 0) {
      tflag = 1;
      iarg += 1;
    } else if (strcmp(args[iarg],"-r") == 0) {
      rflag = 1;
      heffte_set(fft,"reshapeonly",eflag);
      iarg += 1;
    } else if (strcmp(args[iarg],"-o") == 0) {
      oflag = 1;
      iarg += 1;
    } else if (strcmp(args[iarg],"-m") == 0) {
      if (iarg+2 > narg) error_all(syntax);
      mode = atoi(args[iarg+1]);
      iarg += 2;
    } else if (strcmp(args[iarg],"-v") == 0) {
      vflag = 1;
      iarg += 1;
    } else if (strcmp(args[iarg],"-verb") == 0) {
      verb = 1;
      iarg += 1;
    } else error_all(syntax);
  }

  // sanity check on args

  if (N[0] <= 0 || N[1] <= 0 || N[2] <= 0) error_all("Invalid grid size");

  if (proc_i[0] == 0 && proc_i[1] == 0 && proc_i[2] == 0);
  else if (proc_i[0] <= 0 || proc_i[1] <= 0 || proc_i[2] <= 0) error_all("Invalid proc grid");
  else if (proc_i[0]*proc_i[1]*proc_i[2] != nprocs)
    error_all("Specified proc grid does not match nprocs");

  if (proc_o[0] == 0 && proc_o[1] == 0 && proc_o[2] == 0);
  else if (proc_o[0] <= 0 || proc_o[1] <= 0 || proc_o[2] <= 0)
    error_all("Invalid proc grid");
  else if (proc_o[0]*proc_o[1]*proc_o[2] != nprocs)
    error_all("Specified proc grid does not match nprocs");

  if (nloop < 0) error_all("Invalid Nloop");
  if (seed <= 0) error_all("Invalid initialize setting");

if (mode == 0 || mode == 2) permute = 0;
else permute = 2;




if (me == 0){
  printf( "____________________________________________________________________________________________________________________________ \n");
  printf("                                               Testing HEFFTE library                                                         \n");
  printf( "---------------------------------------------------------------------------------------------------------------------------- \n");

  printf("Test summary:\n");
  printf("-------------\n");
  if(mode==0)
  printf("\t%d forward and %d backward 3D-FFTs on %d procs on a complex %dx%dx%d grid\n",nloop,nloop,nprocs,N[0],N[1],N[2]);
  if(mode==1)
  printf("\t%d forward 3D-FFTs on %d procs on a complex %dx%dx%d grid\n",nloop,nloop,nprocs,N[0],N[1],N[2]);
  #if defined(FFT_CUFFTW) || defined(FFT_CUFFT_A) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
    printf("\t1D FFT library       : CUFFT\n");
  #else
    printf("\t1D FFT library       : FFTW3\n");
  #endif
  if(typeid(work)==typeid(double*))
    printf("\tPrecision            : DOUBLE\n");
  if(typeid(work)==typeid(float*))
    printf("\tPrecision            : SINGLE\n");
  if(cflag==POINT)
    printf("\tComunication type    : POINT2POINT\n");
  if(cflag==ALL2ALL)
    printf("\tComunication type    : ALL2ALL\n");
  if(fft->scaled)
    printf("\tScaling after forward: YES\n");
  else
    printf("\tScaling after forward: NO\n");
}

}




/* ----------------------------------------------------------------------
   output FFT grid values
   flag = 0 for initial partition
   flag = 1 for final partition
------------------------------------------------------------------------- */

void heffte_output(int flag, const char *str)
{
  int tmp;

  if (me == 0) printf("%s\n",str);

  for (int iproc = 0; iproc < nprocs; iproc++) {
    if (me != iproc) continue;
    if (me >= 1) MPI_Recv(&tmp,0,MPI_INT,me-1,0,fft_comm,MPI_STATUS_IGNORE);

    int ilocal,jlocal,klocal,iglobal,jglobal,kglobal;

    if (flag == 0) {
      int nxlocal = i_hi[0] - i_lo[0] + 1;
      int nylocal = i_hi[1] - i_lo[1] + 1;

      for (int m = 0; m < nfft_in; m++) {
        ilocal = m % nxlocal;
        jlocal = (m/nxlocal) % nylocal;
        klocal = m / (nxlocal*nylocal);
        iglobal = i_lo[0] + ilocal;
        jglobal = i_lo[1] + jlocal;
        kglobal = i_lo[2] + klocal;
        printf("Value (%d,%d,%d) on proc %d = (%g,%g)\n",
               iglobal,jglobal,kglobal,
               me,work[2*m],work[2*m+1]);
      }
    } else {
      int nxlocal = i_hi[0] - i_lo[0]  + 1;
      int nylocal = i_hi[1] - i_lo[1]  + 1;

      for (int m = 0; m < nfft_in; m++) {
        ilocal = m % nxlocal;
        jlocal = (m/nxlocal) % nylocal;
        klocal = m / (nxlocal*nylocal);
        iglobal = i_lo[0] + ilocal;
        jglobal = i_lo[1] + jlocal;
        kglobal = i_lo[2] + klocal;
        printf("Value (%d,%d,%d) on proc %d = (%g,%g)\n",
               iglobal,jglobal,kglobal,
               me,work[2*m],work[2*m+1]);
      }
    }

    if (me < nprocs-1) MPI_Send(&tmp,0,MPI_INT,me+1,0,fft_comm);
  }
}


/* ----------------------------------------------------------------------
   output timing data
------------------------------------------------------------------------- */

void heffte_timing()
{
  double time1d,time_reshape;
  double time_reshape1,time_reshape2,time_reshape3,time_reshape4;

  // nfft = # of FFTs performed = 2x larger
  int nfft;
  if (mode == 0) nfft = 2*nloop;
  else nfft = nloop;

  double onetime = timefft/nfft;
  double nsize = 1.0 * N[0] * N[1] * N[2];
  double logn = log(nsize);
  double floprate = 5.0 * nsize * logn * 1e-9 / onetime;

  int64_t gridbytes;
  if(typeid(work)==typeid(double*)){
    gridbytes = ((int64_t) sizeof(double)) * 2 * fftsize;
  }
  if(typeid(work)==typeid(float*)){
    gridbytes = ((int64_t) sizeof(float)) * 2 * fftsize;
  }

  if (me == 0) {

    printf("Memory comsuption:\n");
    printf("------------------\n");
    printf("\tMemory usage (per-proc) for FFT grid   = %.2g MB\n",(double) gridbytes / 1024/1024);
    printf("\tMemory usage (per-proc) by FFT library = %.2g MB\n",(double) fft->memusage / 1024/1024);
    printf("\tTotal memory comsuption                = %.2g MB\n", (double) nprocs*fft->memusage / 1024/1024 );

    printf("Processor grids for FFT stages:\n");
    printf("------------------------------- \n");

    if (fft->reshape_preflag && fft->reshape_final_grid){
      printf("\tInitial grid \t1st-direction  \t2nd-direction  \t3rd-direction \tFinal grid \n");
      printf("\t    %d %d %d \t     %d %d %d  \t    %d %d %d \t    %d %d %d \t   %d %d %d \n",
      proc_i[0], proc_i[1], proc_i[2],
      fft->npfast1,fft->npfast2,fft->npfast3,
      fft->npmid1,fft->npmid2,fft->npmid3,
      fft->npslow1,fft->npslow2,fft->npslow3,
      proc_o[0],proc_o[1],proc_o[2]);
    }

    if (!fft->reshape_preflag && fft->reshape_final_grid){
      printf("\t1st-direction  \t2nd-direction  \t3rd-direction \tFinal grid \n");
      printf("\t     %d %d %d  \t    %d %d %d \t    %d %d %d \t   %d %d %d \n",
      fft->npfast1,fft->npfast2,fft->npfast3,
      fft->npmid1,fft->npmid2,fft->npmid3,
      fft->npslow1,fft->npslow2,fft->npslow3,
      proc_o[0],proc_o[1],proc_o[2]);
      printf("\t(Initial grid) \n");
    }

    if (fft->reshape_preflag && !fft->reshape_final_grid){
      printf("\tInitial grid \t1st-direction  \t2nd-direction  \t3rd-direction\n");
      printf("\t    %d %d %d \t     %d %d %d  \t    %d %d %d \t    %d %d %d \n",
      proc_i[0], proc_i[1], proc_i[2],
      fft->npfast1,fft->npfast2,fft->npfast3,
      fft->npmid1,fft->npmid2,fft->npmid3,
      fft->npslow1,fft->npslow2,fft->npslow3);
      printf("\t\t\t\t\t\t\t(Final grid)\n");
    }

    if (!fft->reshape_preflag && !fft->reshape_final_grid){
      printf("\t1st-direction  \t2nd-direction  \t3rd-direction\n");
      printf("\t     %d %d %d  \t    %d %d %d \t    %d %d %d \n",
      fft->npfast1,fft->npfast2,fft->npfast3,
      fft->npmid1,fft->npmid2,fft->npmid3,
      fft->npslow1,fft->npslow2,fft->npslow3);
      printf("\t(Initial grid)\t\t\t (Final grid) \n");
    }

    double computeTime = fft->computeTime/nfft;
    printf( "____________________________________________________________________________________________________________________________\n");
    printf(" ID\tnp \t  nx \t  ny \t  nz \t   Gflops/s  \tExecution (s) \t  One FFT (s) \t Initialisation (s)\t  Max Error \n");
    printf("_3D_\t %d\t%5d\t%5d\t%5d\t%10.4g\t%10.4g\t %10.4g  \t     %10.4g      \t%e\n", nprocs, N[0], N[1], N[2], floprate, computeTime, onetime, timeinit-timesetup, epsmax);
    printf( "---------------------------------------------------------------------------------------------------------------------------- \n");
  }
}
