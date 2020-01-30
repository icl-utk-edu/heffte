/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include <stdlib.h>
#include <stdio.h>
#include "heffte.h"
#include "heffte_common.h"
#include "heffte_reshape3d.h"
#include "mpi.h"

typedef int64_t bigint;

using namespace HEFFTE_NS;


/* ----------------------------------------------------------------------
   Set an internal flag, before setup() or compute()
------------------------------------------------------------------------- */

template <class T>
void heffte_set(FFT3d<T> *fft, const char *keyword, int value)
{
  if (strcmp(keyword,"collective") == 0) fft->collective = value;
  else if (strcmp(keyword,"exchange") == 0) fft->exchange = value;
  else if (strcmp(keyword,"pack") == 0) fft->packflag = value;
  else if (strcmp(keyword,"memflag") == 0) fft->memoryflag = value;
  else if (strcmp(keyword,"reshapeonly") == 0) fft->reshapeonly = value;
  else if (strcmp(keyword,"scale") == 0) fft->scaled = value;
}

template
void heffte_set(FFT3d<double> *fft, const char *keyword, int value);
template
void heffte_set(FFT3d<float> *fft, const char *keyword, int value);

/* ----------------------------------------------------------------------
   Get value of an internal value, return as pointer to value(s)
   caller must cast the pointer correctly to access the value(s)
------------------------------------------------------------------------- */

template <class T>
void *heffte_get(FFT3d<T> *fft, const char *keyword)
{
  if (strcmp(keyword,"fft1d") == 0) return (void *) fft->fft1d;
  else if (strcmp(keyword,"precision") == 0) return (void *) fft->precision;
  else if (strcmp(keyword,"collective") == 0) return &fft->collective;
  else if (strcmp(keyword,"exchange") == 0) return &fft->exchange;
  else if (strcmp(keyword,"pack") == 0) return &fft->packflag;
  else if (strcmp(keyword,"memusage") == 0) return &fft->memusage;
  else if (strcmp(keyword,"npfast1") == 0) return &fft->npfast1;
  else if (strcmp(keyword,"npfast2") == 0) return &fft->npfast2;
  else if (strcmp(keyword,"npfast3") == 0) return &fft->npfast3;
  else if (strcmp(keyword,"npmid1") == 0) return &fft->npmid1;
  else if (strcmp(keyword,"npmid2") == 0) return &fft->npmid2;
  else if (strcmp(keyword,"npmid3") == 0) return &fft->npmid3;
  else if (strcmp(keyword,"npslow1") == 0) return &fft->npslow1;
  else if (strcmp(keyword,"npslow2") == 0) return &fft->npslow2;
  else if (strcmp(keyword,"npslow3") == 0) return &fft->npslow3;
  else if (strcmp(keyword,"npbrick1") == 0) return &fft->npbrick1;
  else if (strcmp(keyword,"npbrick2") == 0) return &fft->npbrick2;
  else if (strcmp(keyword,"npbrick3") == 0) return &fft->npbrick3;
  else if (strcmp(keyword,"ntrial") == 0) return &fft->ntrial;
  else if (strcmp(keyword,"npertrial") == 0) return &fft->npertrial;
  else if (strcmp(keyword,"setuptime") == 0) return &fft->setuptime;
  else if (strcmp(keyword,"cflags") == 0) return fft->cflags;
  else if (strcmp(keyword,"eflags") == 0) return fft->eflags;
  else if (strcmp(keyword,"pflags") == 0) return fft->pflags;
  else if (strcmp(keyword,"tfft") == 0) return fft->tfft;
  else if (strcmp(keyword,"t1d") == 0) return fft->t1d;
  else if (strcmp(keyword,"treshape") == 0) return fft->treshape;
  else if (strcmp(keyword,"treshape1") == 0) return fft->treshape1;
  else if (strcmp(keyword,"treshape2") == 0) return fft->treshape2;
  else if (strcmp(keyword,"treshape3") == 0) return fft->treshape3;
  else if (strcmp(keyword,"treshape4") == 0) return fft->treshape4;
  else return NULL;
}

template
void *heffte_get(FFT3d<double> *fft, const char *keyword);
template
void *heffte_get(FFT3d<float> *fft, const char *keyword);


/* ----------------------------------------------------------------------
   Create plan for performing a Complex-to-Complex 3D FFT
------------------------------------------------------------------------- */
template <class T>
void heffte_plan_create(T *work, FFT3d<T> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                        int permute, int *workspace)
{
  int fftsize,sendsize,recvsize;

  fft->setup(work, N, i_lo, i_hi, o_lo, o_hi,
             permute, fftsize, sendsize, recvsize);

  workspace[0] = fftsize;
  workspace[1] = sendsize;
  workspace[2] = recvsize;
}

template
void heffte_plan_create(double *work, FFT3d<double> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                        int permute, int *workspace);
template
void heffte_plan_create(float *work, FFT3d<float> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                        int permute, int *workspace);


/* ----------------------------------------------------------------------
   Create plan for performing a Real-to-Complex 3D FFT
------------------------------------------------------------------------- */
template <class T>
void heffte_plan_r2c_create(T *work, FFT3d<T> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                            int *workspace)
{
  int fftsize,sendsize,recvsize;

  fft->setup_r2c(work, N, i_lo, i_hi, o_lo, o_hi,
                fftsize, sendsize, recvsize);

  workspace[0] = fftsize;
  workspace[1] = sendsize;
  workspace[2] = recvsize;
}

template
void heffte_plan_r2c_create(double *work, FFT3d<double> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                            int *workspace);
template
void heffte_plan_r2c_create(float *work, FFT3d<float> *fft, int *N, int *i_lo, int *i_hi, int *o_lo, int *o_hi,
                            int *workspace);



/* ----------------------------------------------------------------------
   Pass in user memory for a 3d reshape send/recv
------------------------------------------------------------------------- */

template <class T>
void heffte_setup_memory(FFT3d<T> *fft, T *sendbuf, T *recvbuf)
{
  fft->setup_memory(sendbuf,recvbuf);
}

template
void heffte_setup_memory(FFT3d<double> *fft, double *sendbuf, double *recvbuf);
template
void heffte_setup_memory(FFT3d<float> *fft, float *sendbuf, float *recvbuf);

/* ----------------------------------------------------------------------
   Execute C2C FFT
------------------------------------------------------------------------- */
template <class T>
void heffte_execute(FFT3d<T> *fft, T *data_in, T *data_out, int flag)
{
  if(flag==FORWARD)
    fft->compute(data_in, data_out, 1);
  else if(flag==BACKWARD)
    fft->compute(data_in, data_out, -1);
  else
    error_all("Non valid flag for FFT execution");
}

template
void heffte_execute(FFT3d<double> *fft, double *data_in, double *data_out, int flag);
template
void heffte_execute(FFT3d<float> *fft, float *data_in, float *data_out, int flag);


/* ----------------------------------------------------------------------
   Execute R2C FFT
------------------------------------------------------------------------- */
template <class T>
void heffte_execute_r2c(FFT3d<T> *fft, T *data_in, T *data_out)
{
  fft->compute_r2c(data_in, data_out);
}

template
void heffte_execute_r2c(FFT3d<double> *fft, double *data_in, double *data_out);
template
void heffte_execute_r2c(FFT3d<float> *fft, float *data_in, float *data_out);



/* ----------------------------------------------------------------------
   Perform just the 1d FFTs needed by a 3d FFT, no data movement
------------------------------------------------------------------------- */

template <class T>
void heffte_only_1d_ffts(FFT3d<T> *fft, T *in, int flag)
{
  fft->only_1d_ffts(in,flag);
}

template
void heffte_only_1d_ffts(FFT3d<double> *fft, double *in, int flag);
template
void heffte_only_1d_ffts(FFT3d<float> *fft, float *in, int flag);


/* ----------------------------------------------------------------------
   Perform all the reshapes in a 3d FFT, but no 1d FFTs
------------------------------------------------------------------------- */
template <class T>
void heffte_only_reshapes(FFT3d<T> *fft, T *in, T *out, int flag)
{
  fft->only_reshapes(in,out,flag);
}

template
void heffte_only_reshapes(FFT3d<double> *fft, double *in, double *out, int flag);
template
void heffte_only_reshapes(FFT3d<float> *fft, float *in, float *out, int flag);


/* ----------------------------------------------------------------------
   Perform just a single 3d reshape operation
------------------------------------------------------------------------- */
template <class T>
void heffte_only_one_reshape(FFT3d<T> *fft, T *in, T *out, int flag, int which)
{
  fft->only_one_reshape(in,out,flag,which);
}

template
void heffte_only_one_reshape(FFT3d<double> *fft, double *in, double *out, int flag, int which);
template
void heffte_only_one_reshape(FFT3d<float> *fft, float *in, float *out, int flag, int which);


/* ----------------------------------------------------------------------
// Initialisation
/* ----------------------------------------------------------------------
   simple Park RNG
   pass in non-zero seed
------------------------------------------------------------------------- */
double random_init(int &seed)
{
  int k = seed/IQ;
  seed = IA*(seed-k*IQ) - IR*k;
  if (seed < 0) seed += IM;
  double ans = AM*seed;
  return ans;
}

template <class T>
void heffte_initialize_host(T *work, int n, int seed, int data_type)
{
    for (int i = 0; i < 2*n; i++){
      work[i] = random_init(seed);
      if ( data_type == 0 && i >= n)
        work[i] = 0;
    }

    if( (data_type != 0) && (data_type != 1)) error_all("Not valid initialization!");
}

template
void heffte_initialize_host(double *work, int n, int seed, int data_type);
template
void heffte_initialize_host(float *work, int n, int seed, int data_type);


/* ----------------------------------------------------------------------
   Error validation function
------------------------------------------------------------------------- */

template <class T>
void heffte_validate(T* work, int n, int seed, double &epsmax, MPI_Comm world)
{
  double delta;
  double epsilon = 0.0;
  double newvalue;

  for (int i = 0; i < 2*n; i++) {
    newvalue = random_init(seed);
    delta = fabs(work[i]-newvalue);
    if (delta > epsilon) epsilon = delta;
  }
  MPI_Allreduce(&epsilon,&epsmax,1,MPI_DOUBLE,MPI_MAX,world);
}

template
void heffte_validate(double* work, int n, int seed, double &epsmax, MPI_Comm world);
template
void heffte_validate(float* work, int n, int seed, double &epsmax, MPI_Comm world);

// TODO: update error handlers
void error_all(const char *str)
{
  MPI_Barrier(MPI_COMM_WORLD);
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  if (me == 0) printf("ERROR: %s\n",str);
  MPI_Finalize();
  exit(1);
}

void error_one(const char *str)
{
  int me;
  MPI_Comm_rank(MPI_COMM_WORLD,&me);
  printf("ERROR on proc %d: %s\n",me,str);
  MPI_Abort(MPI_COMM_WORLD,1);
}


// Grid processor
/* ----------------------------------------------------------------------
   Partition FFT grid
   once for input grid, once for output grid
   use Px,Py,Pz for in/out
------------------------------------------------------------------------- */

void heffte_grid_setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                       int* proc_i, int* proc_o, int me, int &nfft_in, int &nfft_out)
{
  // ipx,ipy,ipz = my position in input 3d grid of procs

  int ipx = me % proc_i[0];
  int ipy = (me/proc_i[0]) % proc_i[1];
  int ipz = me / (proc_i[0]*proc_i[1]);

  // nlo,nhi = lower/upper limits of the 3d brick I own

  i_lo[0] = static_cast<int> (1.0 * ipx * N[0] / proc_i[0]);
  i_hi[0] = static_cast<int> (1.0 * (ipx+1) * N[0] / proc_i[0]) - 1;

  i_lo[1] = static_cast<int> (1.0 * ipy * N[1] / proc_i[1]);
  i_hi[1] = static_cast<int> (1.0 * (ipy+1) * N[1] / proc_i[1]) - 1;

  i_lo[2] = static_cast<int> (1.0 * ipz * N[2] / proc_i[2]);
  i_hi[2] = static_cast<int> (1.0 * (ipz+1) * N[2] / proc_i[2]) - 1;

  nfft_in = (i_hi[0]-i_lo[0]+1) * (i_hi[1]-i_lo[1]+1) * (i_hi[2]-i_lo[2]+1);

// printf("in %d,%d,%d,%d,%d, %d\n", i_lo[0], i_hi[0], i_lo[1], i_hi[1], i_lo[2], i_hi[2]);

  // ipx,ipy,ipz = my position in output 3d grid of procs

  ipx = me % proc_o[0];
  ipy = (me/proc_o[0]) % proc_o[1];
  ipz = me / (proc_o[0]*proc_o[1]);

  // nlo,nhi = lower/upper limits of the 3d brick I own

  o_lo[0] = static_cast<int> (1.0 * ipx * N[0] / proc_o[0]);
  o_hi[0] = static_cast<int> (1.0 * (ipx+1) * N[0] / proc_o[0]) - 1;

  o_lo[1] = static_cast<int> (1.0 * ipy * N[1] / proc_o[1]);
  o_hi[1] = static_cast<int> (1.0 * (ipy+1) * N[1] / proc_o[1]) - 1;

  o_lo[2] = static_cast<int> (1.0 * ipz * N[2] / proc_o[2]);
  o_hi[2] = static_cast<int> (1.0 * (ipz+1) * N[2] / proc_o[2]) - 1;

  // printf("out %d,%d,%d,%d,%d, %d\n", o_lo[0], o_hi[0], o_lo[1], o_hi[1], o_lo[2], o_hi[2]);

  nfft_out = (o_hi[0]-o_lo[0]+1) * (o_hi[1]-o_lo[1]+1) * (o_hi[2]-o_lo[2]+1);
}




void heffte_proc_setup(int *N, int *proc_grid, int nprocs)
{
    if (proc_grid[0] != 0 || proc_grid[1] != 0 || proc_grid[2] != 0) return;
      heffte_proc3d(N,proc_grid[0],proc_grid[1],proc_grid[2],nprocs);
}




void heffte_proc3d(int *N, int &px, int &py, int &pz, int nprocs)
{
  int ipx,ipy,ipz,nremain;
  double boxx,boxy,boxz,surf;
  double xprd = N[0];
  double yprd = N[1];
  double zprd = N[2];

  double bestsurf = 2.0 * (xprd*yprd + yprd*zprd + zprd*xprd);
  ipx = 1;
  while (ipx <= nprocs) {
    if (nprocs % ipx == 0) {
      nremain = nprocs/ipx;
      ipy = 1;
      while (ipy <= nremain) {
        if (nremain % ipy == 0) {
          ipz = nremain/ipy;
          boxx = xprd/ipx;
          boxy = yprd/ipy;
          boxz = zprd/ipz;
          surf = boxx*boxy + boxy*boxz + boxz*boxx;
          if (surf < bestsurf) {
            bestsurf = surf;
            px = ipx;
            py = ipy;
            pz = ipz;
          }
        }
        ipy++;
      }
    }
    ipx++;
  }

  if (px*py*pz != nprocs)
    error_all("Computed proc grid does not match nprocs");
}


// Memory allocation
template <class T>
void heffte_allocate(heffte_memory_type_t mem, T **work, int fftsize, int64_t &nbytes)
{
  heffte_memory_type_t mem_type(mem);

  if ((1 << mem_type) & mem_aligned)
    nbytes = ((bigint) sizeof(T)) * (2*fftsize + FFT_MEMALIGN);
  else
    nbytes = ((bigint) sizeof(T)) * 2*fftsize;

  class Memory *memory;
  memory = new Memory();
  *work = (T *) memory->smalloc(nbytes, mem_type);
  delete memory;

  if (nbytes && work == NULL) error_one("Failed malloc for FFT grid");
  nbytes = ((bigint) sizeof(T)) * ( 2*fftsize );
}

template
void heffte_allocate(heffte_memory_type_t mem, double **work, int fftsize, int64_t &nbytes);
template
void heffte_allocate(heffte_memory_type_t mem, float **work, int fftsize, int64_t &nbytes);


// Memory deallocation
template <class T>
void heffte_deallocate(heffte_memory_type_t mem, T *ptr)
{
  heffte_memory_type_t mem_type(mem);

  class Memory *memory;
  memory = new Memory();
      memory->sfree(ptr, mem_type);
  delete memory;
}

template
void heffte_deallocate(heffte_memory_type_t mem, double *ptr);
template
void heffte_deallocate(heffte_memory_type_t mem, float *ptr);



/**
 * Initialize HEFFTE library.
 * @return 0 if successful.
 */
int heffte_init(){
	return 0;
}

/**
 * Initializes the library.
 * @param nthreads The number of OpenMP threads to use for execution of local FFT.
 * @return 0 if successful
 */
int heffte_init(int nthreads) {
	int threads_ok = 1;
#ifdef FFT_FFTW3_OMP
	if (threads_ok)
		threads_ok = fftw_init_threads();
	if (threads_ok)
		fftw_plan_with_nthreads(nthreads);
#endif
	return (!threads_ok);
}

/**
 * Cleanup all CPU resources
 */
void heffte_cleanup() {
  #ifdef FFT_FFTW3_OMP
	fftw_cleanup_threads();
	fftw_cleanup();
  #endif
}
