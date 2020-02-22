/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/
#ifndef TESTING_H
#define TESTING_H

// includes, system
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>
#include <typeinfo>
#include <mpi.h>

// includes, project
#include "heffte_common.h"
#include "heffte_fft3d.h"
#include "heffte_trace.h"
#include "heffte.h"

/***************************************************************************//**
 *  For portability to Windows
 */
#if defined( _WIN32 ) || defined( _WIN64 )
#endif

/***************************************************************************//**
 *  For Intel compiler
 */

#if defined(__INTEL_COMPILER__)
  #ifndef FFT_INTEL_NO_TBB
    #define FFT_USE_TBB_ALLOCATOR
    #include "tbb/scalable_allocator.h"
  #else
    #include <malloc.h>
  #endif
#endif

typedef int heffte_int_t;

/* ----------------------------------------------------------------------
Enumerating options for flags
------------------------------------------------------------------------- */
enum{POINT,ALL2ALL,COMBO};
enum{PENCIL,BRICK};
enum{ARRAY,POINTER,MEMCPY};

/***************************************************************************//**
 *  Class to handle heFFTe variables
 */
class heffte_opts {

  public:
// MPI parameters
  MPI_Comm fft_comm;
  int me, nprocs;

// constructor
  heffte_opts(MPI_Comm default_comm = MPI_COMM_WORLD){
    fft_comm = default_comm;
    MPI_Comm_size(fft_comm, &nprocs);
    MPI_Comm_rank(fft_comm, &me);
  }

// methods
  template <class T> void parse_opts( int argc, char** argv, FFT3d<T> *fft);
  template <class T> void heffte_timing(FFT3d<T> *fft);
  template <class T> void heffte_print_grid(int flag, const char *str, T *work, heffte_int_t nfft_in,
                                      int *i_lo, int * i_hi, int *o_lo, int * o_hi);

// variables
  heffte_int_t N[3];
  heffte_int_t proc_i[3];
  heffte_int_t proc_o[3];
  int nloop;
  int mode;
  int permute;
  int seed;

  heffte_int_t workspace[3];
  // workspace[0] = size of FFT, returned by setup()
  // workspace[1] = size of sending buffer needed for transposition
  // workspace[2] = size of receiving buffer needed for transposition

// flags
  int oflag, cflag, eflag, pflag, mflag, rflag, sflag, tflag, vflag, verb;

// Timing array
  double  timeinit, timeplan, timefft; // Basic timing: time_initialization, time for plan creation, time for computation
  double  epsmax = 0.0;
  int64_t nbytes;

// constants
  const char *syntax =
    "Example of correct syntax: test3d_exec -g Nx Nx Nz -pin Px Py Pz -pout Qx Qy Qz -n Nloop -i 82783 \n"
    "                           -c point/all/combo -e pencil/brick -p array/ptr/memcpy\n"
    "                           -t -r -o -v";

};


/* ----------------------------------------------------------------------
  Detailed timing vector, see documentation: timing.md
------------------------------------------------------------------------- */
double timing_array[NTIMING_VARIABLES];

/* ----------------------------------------------------------------------
  Parse user options from command line
------------------------------------------------------------------------- */
template <class T>
void heffte_opts::parse_opts( int argc, char** argv, FFT3d<T> *fft)
{

// defaults values
  N[0] = N[1] = N[2] = 8;
  proc_i[0] = proc_i[1] = proc_i[2] = 0;
  proc_o[0] = proc_o[1] = proc_o[2] = 0;

  cflag = COMBO;   // communication flag
  eflag = PENCIL;  // data exchange flag
  pflag = MEMCPY;  // packing flag

  nloop = 1;       // # of loops to execute FFT
  mflag = 1;       // library allocates memory internally

// Flags below are set as inactive by default
  rflag = 0;       // remaponly flag
  sflag = 0;       // scale after forward FFT computation
  tflag = 0;       // tuning flag
  oflag = 0;       // grid values printing flag
  mode  = 0;       // FFT mode selection
  vflag = 0;       // Error validation flag
  verb  = 0;       // Print architecture charateristics

// Get MPI information from communicator
  int me, nprocs;
  MPI_Comm_size(fft->world, &nprocs);
  MPI_Comm_rank(fft->world, &me);

// read from command line
  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-h") == 0) {
      error_all(syntax);
    } else if (strcmp(argv[iarg],"-g") == 0) {
      if (iarg+4 > argc) error_all(syntax);
      N[0] = atoi(argv[iarg+1]);
      N[1] = atoi(argv[iarg+2]);
      N[2] = atoi(argv[iarg+3]);
      iarg += 4;
    } else if (strcmp(argv[iarg],"-pin") == 0) {
      if (iarg+4 > argc) error_all(syntax);
      proc_i[0] = atoi(argv[iarg+1]);
      proc_i[1] = atoi(argv[iarg+2]);
      proc_i[2] = atoi(argv[iarg+3]);
      iarg += 4;
    } else if (strcmp(argv[iarg],"-pout") == 0) {
      if (iarg+4 > argc) error_all(syntax);
      proc_o[0] = atoi(argv[iarg+1]);
      proc_o[1] = atoi(argv[iarg+2]);
      proc_o[2] = atoi(argv[iarg+3]);
      iarg += 4;
    } else if (strcmp(argv[iarg],"-n") == 0) {
      if (iarg+2 > argc) error_all(syntax);
      nloop = atoi(argv[iarg+1]);
      iarg += 2;
    } else if (strcmp(argv[iarg],"-i") == 0) {
      if (iarg+2 > argc) error_all(syntax);
        seed = atoi(argv[iarg+1]) + me;
      iarg += 2;
    } else if (strcmp(argv[iarg],"-c") == 0) {
      if (iarg+2 > argc) error_all(syntax);
      if (strcmp(argv[iarg+1],"point") == 0) cflag = POINT;
      else if (strcmp(argv[iarg+1],"all") == 0) cflag = ALL2ALL;
      else if (strcmp(argv[iarg+1],"combo") == 0) cflag = COMBO;
      else error_all(syntax);
      heffte_set(fft,"collective",cflag);
      iarg += 2;
    } else if (strcmp(argv[iarg],"-e") == 0) {
      if (iarg+2 > argc) error_all(syntax);
      if (strcmp(argv[iarg+1],"pencil") == 0) eflag = PENCIL;
      else if (strcmp(argv[iarg+1],"brick") == 0) eflag = BRICK;
      else error_all(syntax);
      heffte_set(fft,"exchange",eflag);
      iarg += 2;
    } else if (strcmp(argv[iarg],"-p") == 0) {
      if (iarg+2 > argc) error_all(syntax);
      if (strcmp(argv[iarg+1],"array") == 0) pflag = ARRAY;
      else if (strcmp(argv[iarg+1],"ptr") == 0) pflag = POINTER;
      else if (strcmp(argv[iarg+1],"memcpy") == 0) pflag = MEMCPY;
      else error_all(syntax);
      heffte_set(fft,"pack",pflag);
      iarg += 2;
    } else if (strcmp(argv[iarg],"-r") == 0) {
      rflag = 1;
      heffte_set(fft,"reshapeonly",eflag);
      iarg += 1;
    } else if (strcmp(argv[iarg],"-s") == 0) {
      sflag = 1;
      iarg += 1;
    } else if (strcmp(argv[iarg],"-t") == 0) {
      tflag = 1;
      iarg += 1;
    }  else if (strcmp(argv[iarg],"-o") == 0) {
      oflag = 1;
      iarg += 1;
    } else if (strcmp(argv[iarg],"-m") == 0) {
      if (iarg+2 > argc) error_all(syntax);
      mode = atoi(argv[iarg+1]);
      iarg += 2;
    } else if (strcmp(argv[iarg],"-v") == 0) {
      vflag = 1;
      iarg += 1;
    } else if (strcmp(argv[iarg],"-verb") == 0) {
      verb = 1;
      iarg += 1;
    } else error_all(syntax);
  }

  // sanity check on argv
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


// Set user parameters

  heffte_set(fft,"collective", cflag);
  heffte_set(fft,"exchange", eflag);
  heffte_set(fft,"pack", pflag);
  heffte_set(fft,"memflag", mflag);
  heffte_set(fft,"reshapeonly", rflag);
  heffte_set(fft,"scale", sflag);

// Display user selections
if (me == 0){
  printf( "____________________________________________________________________________________________________________________________ \n");
  printf("                                               Testing HEFFTE library                                                         \n");
  printf( "---------------------------------------------------------------------------------------------------------------------------- \n");

  printf("Test summary:\n");
  printf("-------------\n");
  char *fft_type;
    printf("\tComputation of 3D FFT \n");

  if(mode==0)
    printf("\t%d forward and %d backward 3D-FFTs on %d procs on a %dx%dx%d grid\n",nloop,nloop,nprocs,N[0],N[1],N[2]);
  if(mode==1)
    printf("\t%d forward 3D-FFTs on %d procs on a complex %dx%dx%d grid\n",nloop,nprocs,N[0],N[1],N[2]);

  #if defined(FFT_CUFFTW) || defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
    printf("\t1D FFT library       : CUFFT\n");
  #else
    printf("\t1D FFT library       : FFTW3\n");
  #endif

  if(typeid(T) == typeid(double))
    printf("\tPrecision            : DOUBLE\n");
  if(typeid(T) == typeid(float))
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
  Print detailed timing of kernels
------------------------------------------------------------------------- */
template <class T>
void heffte_opts::heffte_timing(FFT3d<T> *fft)
{
// Get MPI information from communicator
  int me, nprocs;
  MPI_Comm_size(fft->world, &nprocs);
  MPI_Comm_rank(fft->world, &me);

  int nfft; // # of FFTs performed
  if (mode == 0) nfft = 2*nloop;
  else nfft = nloop;

  double onetime = timefft/nfft;
  double nsize = 1.0 * N[0] * N[1] * N[2];
  double log2n = log(nsize)/log(2.0);
  double floprate = 5.0 * nsize * log2n * 1e-9 / onetime;

  int64_t gridbytes;
  if(typeid(T) == typeid(double)){
    gridbytes = ((int64_t) sizeof(double)) * 2 * workspace[0];
  }
  if(typeid(T) == typeid(float)){
    gridbytes = ((int64_t) sizeof(float)) * 2 * workspace[0];
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

    printf( "____________________________________________________________________________________________________________________________\n");
    printf(" ID\tnp \t  nx \t  ny \t  nz \t   Gflops/s \t  One FFT (s) \t Initialisation (s)\t  Max Error \n");
    printf("_3D_\t %d\t%5d\t%5d\t%5d\t%10.4g\t %10.4g  \t     %10.4g      \t%e\n", nprocs, N[0], N[1], N[2], floprate, onetime, timeinit, epsmax);
    printf( "---------------------------------------------------------------------------------------------------------------------------- \n");
    printf(" \t   me\t  compute\t  perform  \t     pack \t    unpack\t    scale\t    A2A\t         MPI_total \n");
  }

  if(me ==0){ // Comment for detailed data per processor
    // Get total MPI time
    timing_array[6] = timing_array[0] - (timing_array[1] + timing_array[2] + timing_array[3] + timing_array[4]);
    printf("_Timing_    %d\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g\t%10.4g \n", me, timing_array[0], timing_array[1], timing_array[2], timing_array[3],
    timing_array[4], timing_array[5], timing_array[6]);
    printf( "---------------------------------------------------------------------------------------------------------------------------- \n");
  }

}


/* ----------------------------------------------------------------------
   Print FFT grid values
   flag = 0 for initial partition
   flag = 1 for final partition
------------------------------------------------------------------------- */
template <class T>
void heffte_opts::heffte_print_grid(int flag, const char *str, T *work, heffte_int_t nfft_in,
                                    int *i_lo, int * i_hi, int *o_lo, int * o_hi)
{
  int tmp;
  printf("%s\n",str);

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
               iglobal, jglobal, kglobal,
               me, work[2*m], work[2*m+1]);
        if((m%4)==3) printf("\n");
    }
  } else {
      int nxlocal = o_hi[0] - o_lo[0] + 1;
      int nylocal = o_hi[1] - o_lo[1] + 1;

      for (int m = 0; m < nfft_in; m++) {
        ilocal = m % nxlocal;
        jlocal = (m/nxlocal) % nylocal;
        klocal = m / (nxlocal*nylocal);
        iglobal = o_lo[0] + ilocal;
        jglobal = o_lo[1] + jlocal;
        kglobal = o_lo[2] + klocal;
        printf("Value (%d,%d,%d) on proc %d = (%g,%g)\n",
               iglobal, jglobal, kglobal,
               me, work[2*m], work[2*m+1]);
         if((m%4)==3) printf("\n");
      }
    }

    if (me < nprocs-1) MPI_Send(&tmp,0,MPI_INT,me+1,0,fft_comm);
  }
}

#endif /* TESTING_H */
