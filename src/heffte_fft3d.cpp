/**
 * @class
 * CPU functions of HEFFT
 */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "heffte_old_api_cuda.h"
#include "heffte_fft3d.h"
#include "heffte_scale.h"
#include "heffte_trace.h"

#ifdef Heffte_ENABLE_FFTW

using namespace HEFFTE;

#define BIG 1.0e20

typedef int64_t bigint;


/*! \fn
 * instantiate a 3d FFT
 * @param user_comm  MPI communicator for the P procs which own the data
 */
template <class U>
FFT3d<U>::FFT3d(MPI_Comm user_comm)
{
  world = user_comm;
  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // default settings
  // user must change them before setup()

  collective = 2;
  exchange = 0;
  packflag = 2;
  memoryflag = 1;

  // default settings
  // user can change them before compute()

  scaled = 1;
  reshapeonly = 0;

  // tuning results

  ntrial = npertrial = 0;
  cbest = ebest = pbest = -1;
  besttime = 0.0;

  // Memory and Error classes

  memory = new Memory();

  error = new Error(world);

  // allowed prime factors for each FFT grid dimension

  primes = {2, 3, 5};

  // initialize memory allocations

  reshape_prefast = reshape_fastmid = reshape_midslow = reshape_postslow = NULL;
  reshape_preslow = reshape_slowmid = reshape_midfast = reshape_postfast = NULL;
  fft_fast = fft_mid = fft_slow = NULL;

  memusage = 0;
  sendbuf = recvbuf = NULL;

  setupflag = 0;
  setupflag_r2c = 0;
  setup_memory_flag = 0;
}

template
FFT3d<double>::FFT3d(MPI_Comm user_comm);
template
FFT3d<float>::FFT3d(MPI_Comm user_comm);

/* ----------------------------------------------------------------------
   delete a 3d FFT
------------------------------------------------------------------------- */
template <class U>
FFT3d<U>::~FFT3d()
{
  delete memory;
  delete error;

  if (setupflag) deallocate_setup();
  if (setupflag_r2c) deallocate_setup_r2c();

  if (memoryflag) deallocate_setup_memory();
}

template
FFT3d<double>::~FFT3d();
template
FFT3d<float>::~FFT3d();



/**
 * Create and setup plan for performing a 3D FFT
 * @param N Integer array of size 3, corresponding to the global data size
 * @param i_lo Integer array of size 3, lower-input bounds of data I own on each of 3 directions
 * @param i_hi Integer array of size 3, upper-input bounds of data I own on each of 3 directions
 * @param o_lo Integer array of size 3, lower-input bounds of data I own on each of 3 directions
 * @param o_hi Integer array of size 3, upper-input bounds of data I own on each of 3 directions
 * @param user_permute Permutation in storage order of indices on output
 * @return user_fftsize = Size of in/out FFT arrays required from caller
 * @return user_sendsize = Size of send buffer, caller may choose to provide it
 * @return user_recvsize = Size of recv buffer, caller may choose to provide it
 */

template <class U>
void FFT3d<U>::setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                  int user_permute, int &user_fftsize, int &user_sendsize, int &user_recvsize)
{
  int flag,allflag;
  memory->memory_type = mem_type;  // Assign type of memory for fft variables

  if (setupflag) error->all("FFT C2C is already setup");
  setupflag = 1;

  // internal copies of input params

  nfast = N[0];
  nmid  = N[1];
  nslow = N[2];

  in_ilo = i_lo[0]; in_ihi = i_hi[0];
  in_jlo = i_lo[1]; in_jhi = i_hi[1];
  in_klo = i_lo[2]; in_khi = i_hi[2];
  out_ilo = o_lo[0]; out_ihi = o_hi[0];
  out_jlo = o_lo[1]; out_jhi = o_hi[1];
  out_klo = o_lo[2]; out_khi = o_hi[2];

  permute = user_permute;

  // all dimensions must be >= 2
  // all dimensions must be factorable

  if (nfast < 2 || nmid < 2 || nslow < 2)
    error->all("Each FFT dimension must be >= 2");

  if (!prime_factorable(nfast)) error->all("Invalid nfast");
  if (!prime_factorable(nmid)) error->all("Invalid nmid");
  if (!prime_factorable(nslow)) error->all("Invalid nslow");

  // set collective flags for different reshape operations
  // bp = brick2pencil or pencil2brick, pp = pencel2pencil

  if (collective == 0) collective_bp = collective_pp = 0;
  else if (collective == 1) collective_bp = collective_pp = 1;
  else {
    collective_bp = 0;
    collective_pp = 1;
  }

  // inout_layout_same is set only if:
  // in/out indices are same on every proc and permute = 0

  flag = 0;
  if (in_ilo != out_ilo || in_ihi != out_ihi ||
      in_jlo != out_jlo || in_jhi != out_jhi ||
      in_klo != out_klo || in_khi != out_khi) flag = 1;
  if (permute) flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);
  if (allflag) inout_layout_same = 0;
  else inout_layout_same = 1;

  // compute partitioning of FFT grid across procs for each pencil layout
  // if exchange set, also partition in 3d for brick layout
  // np = # of procs in each dimension
  // ip = my location in each dimension

  factor(nprocs);

  procfactors(1, nmid, nslow,
              npfast1, npfast2, npfast3, ipfast1, ipfast2, ipfast3);
  procfactors(nfast, 1, nslow,
              npmid1, npmid2, npmid3, ipmid1, ipmid2, ipmid3);
  procfactors(nfast, nmid, 1,
              npslow1, npslow2, npslow3, ipslow1, ipslow2, ipslow3);

  if (exchange)
    procfactors(nfast, nmid, nslow,
                npbrick1, npbrick2, npbrick3, ipbrick1, ipbrick2, ipbrick3);
  else npbrick1 = npbrick2 = npbrick3 = 0;

  // reshape from initial layout to fast pencil layout
  // reshape_preflag = 1 if reshape is needed, else 0
  // not needed if all procs own entire fast dimension initially
  // fast indices = data layout before/after 1st set of FFTs

  if (in_ilo == 0 && in_ihi == nfast-1) flag = 0;
  else flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);

  if (allflag == 0) {
    reshape_preflag = 0;
    fast_ilo = in_ilo;
    fast_ihi = in_ihi;
    fast_jlo = in_jlo;
    fast_jhi = in_jhi;
    fast_klo = in_klo;
    fast_khi = in_khi;
  } else {
    reshape_preflag = 1;
    fast_ilo = 0;
    fast_ihi = nfast - 1;
    fast_jlo = ipfast2*nmid/npfast2;
    fast_jhi = (ipfast2+1)*nmid/npfast2 - 1;
    fast_klo = ipfast3*nslow/npfast3;
    fast_khi = (ipfast3+1)*nslow/npfast3 - 1;
  }

  // reshape from fast pencil layout to mid pencil layout
  // always needed, b/c permutation changes
  // mid indices = data layout before/after 2nd set of FFTs

  mid_ilo = ipmid1*nfast/npmid1;
  mid_ihi = (ipmid1+1)*nfast/npmid1 - 1;
  mid_jlo = 0;
  mid_jhi = nmid - 1;
  mid_klo = ipmid3*nslow/npmid3;
  mid_khi = (ipmid3+1)*nslow/npmid3 - 1;

  // reshape from mid pencil layout to slow pencil layout
  // always needed, b/c permutation changes
  // slow indices = data layout before/after 3rd set of FFTs
  // if final layout is slow pencil with permute=2, set slow = out

  if (permute == 2 && out_klo == 0 && out_khi == nslow-1) flag = 0;
  else flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);

  if (allflag == 0) {
    slow_ilo = out_ilo;
    slow_ihi = out_ihi;
    slow_jlo = out_jlo;
    slow_jhi = out_jhi;
    slow_klo = out_klo;
    slow_khi = out_khi;
  } else {
    slow_ilo = ipslow1*nfast/npslow1;
    slow_ihi = (ipslow1+1)*nfast/npslow1 - 1;
    slow_jlo = ipslow2*nmid/npslow2;
    slow_jhi = (ipslow2+1)*nmid/npslow2 - 1;
    slow_klo = 0;
    slow_khi = nslow - 1;
  }


// Check if reshape is needed from 3rd-direction to final grid, regardless permutation
  if (out_ilo == slow_ilo && out_ihi == slow_ihi &&
      out_jlo == slow_jlo && out_jhi == slow_jhi &&
      out_klo == slow_klo && out_khi == slow_khi) flag = 0;
  else flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);

  if (allflag == 0){
     reshape_final_grid = 0;
  }
  else {
    reshape_final_grid = 1;
  }

  // reshape from slow pencil layout to final layout
  // reshape_postflag = 1 if reshape is needed, else 0
  // not needed if permute=2 and slow = out already

  if (permute == 2 &&
      out_ilo == slow_ilo && out_ihi == slow_ihi &&
      out_jlo == slow_jlo && out_jhi == slow_jhi &&
      out_klo == slow_klo && out_khi == slow_khi) flag = 0;

  else flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);

  if (allflag == 0){
     reshape_postflag = 0;
  }
  else {
    reshape_postflag = 1;
  }

  // if exchange is set, then reshape for fast/mid and mid/slow
  // reshape will be two stages, with brick layout and brick indices inbetween

  if (exchange) {
    brick_ilo = ipbrick1*nfast/npbrick1;
    brick_ihi = (ipbrick1+1)*nfast/npbrick1 - 1;
    brick_jlo = ipbrick2*nmid/npbrick2;
    brick_jhi = (ipbrick2+1)*nmid/npbrick2 - 1;
    brick_klo = ipbrick3*nslow/npbrick3;
    brick_khi = (ipbrick3+1)*nslow/npbrick3 - 1;
  }

  // create Reshape instances for 4 forward reshapes
  // likewise for inverse reshapes if in/out layout is not the same
  // create calls return max size of send/recv buffers needed by reshapes

  sendsize = recvsize = 0;
  reshape_forward_create(sendsize, recvsize);
  if (!inout_layout_same) reshape_inverse_create(sendsize, recvsize);

  // insize/outsize = # of FFT data points in initial/final layout
  // fastsize/midsize/slowsize = # of data points in fast/mid/slow layout
  // maxsize = max of all these sizes, returned to caller

  insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) *
    (in_khi-in_klo+1);
  outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
    (out_khi-out_klo+1);

  fastsize = (fast_ihi-fast_ilo+1) * (fast_jhi-fast_jlo+1) *
    (fast_khi-fast_klo+1);
  midsize = (mid_ihi-mid_ilo+1) * (mid_jhi-mid_jlo+1) *
    (mid_khi-mid_klo+1);
  slowsize = (slow_ihi-slow_ilo+1) * (slow_jhi-slow_jlo+1) *
    (slow_khi-slow_klo+1);
  if (exchange)
    bricksize = (brick_ihi-brick_ilo+1) * (brick_jhi-brick_jlo+1) *
      (brick_khi-brick_klo+1);

  fftsize = std::max(insize, outsize);
  fftsize = std::max(fftsize, fastsize);
  fftsize = std::max(fftsize, midsize);
  fftsize = std::max(fftsize, slowsize);
  if (exchange) fftsize = std::max(fftsize,bricksize);

  // setup for 3 sets of 1d FFTs, also scaling normalization
  // outsize must be already set for setup_ffts() to use to setup scaling
  // norm must allow for nfast*nmid*nslow to exceed a 4-byte int (2B)

  fft_fast = new FFT1d;
  fft_mid = new FFT1d;
  fft_slow = new FFT1d;

  fft_fast->length = nfast;
  fft_fast->n = (fast_jhi-fast_jlo+1) * (fast_khi-fast_klo+1);
  fft_fast->total = fft_fast->n * fft_fast->length;

  fft_mid->length = nmid;
  fft_mid->n = (mid_ihi-mid_ilo+1) * (mid_khi-mid_klo+1);
  fft_mid->total = fft_mid->n * fft_mid->length;

  fft_slow->length = nslow;
  fft_slow->n = (slow_ihi-slow_ilo+1) * (slow_jhi-slow_jlo+1);
  fft_slow->total = fft_slow->n * fft_slow->length;

  setup_ffts();

  norm = 1.0/((bigint) nfast * nmid*nslow);
  normnum = outsize;

  // allocate sendbuf, recvbuf arrays to max sizes needed by any reshape

  if (memoryflag) {
    setup_memory_flag = 1;
    if (sendsize) {
      sendbuf = (U *) memory->smalloc(sendsize*sizeof(U), mem_type);
      if (!sendbuf) error->one("Could not allocate sendbuf array");
    }
    if (recvsize) {
      recvbuf = (U *) memory->smalloc(recvsize*sizeof(U), mem_type);
      if (!recvbuf) error->one("Could not allocate recvbuf array");
    }
  }

  // return buffer sizes to caller

  user_fftsize  = fftsize;
  user_sendsize = sendsize;
  user_recvsize = recvsize;

  // set memusage for FFT and Reshape memory

  memusage = 0;

  if (memoryflag) {
    memusage += (int64_t) sendsize * sizeof(U);
    memusage += (int64_t) recvsize * sizeof(U);
  }

  memusage += reshape_memory();

}

template
void FFT3d<double>::setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                  int user_permute, int &user_fftsize, int &user_sendsize, int &user_recvsize);
template
void FFT3d<float>::setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                  int user_permute, int &user_fftsize, int &user_sendsize, int &user_recvsize);



/* ----------------------------------------------------------------------
  Real to Complex FFT, setting up function
------------------------------------------------------------------------- */
template <class U>
void FFT3d<U>::setup_r2c(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                   int &user_fftsize, int &user_sendsize, int &user_recvsize)
{
  int flag,allflag;
  memory->memory_type = mem_type;  // Assign type of memory for fft variables

  if (setupflag_r2c) error->all("FFT R2C is already setup");
  if (setupflag) error->all("FFT C2C is already setup");

  setupflag_r2c = 1;

  // internal copies of input params

  nfast = N[0];
  nmid  = N[1];
  nslow = N[2];

  in_ilo = i_lo[0]; in_ihi = i_hi[0];
  in_jlo = i_lo[1]; in_jhi = i_hi[1];
  in_klo = i_lo[2]; in_khi = i_hi[2];
  out_ilo = o_lo[0]; out_ihi = o_hi[0];
  out_jlo = o_lo[1]; out_jhi = o_hi[1];
  out_klo = o_lo[2]; out_khi = o_hi[2];

  // all dimensions must be >= 2
  // all dimensions must be factorable
  if (nfast < 2 || nmid < 2 || nslow < 2)
    error->all("Each FFT dimension must be >= 2");

  if (!prime_factorable(nfast)) error->all("Invalid nfast");
  if (!prime_factorable(nmid)) error->all("Invalid nmid");
  if (!prime_factorable(nslow)) error->all("Invalid nslow");


  // compute partitioning of FFT grid across procs for each pencil layout

  factor(nprocs);

  // get grid for fast direction
  procfactors(1, nmid, nslow,
            npfast1, npfast2, npfast3, ipfast1, ipfast2, ipfast3);

  if (in_ilo == 0 && in_ihi == nfast-1) flag = 0;
  else flag = 1;
  MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);


// For R2C, Y and Z directions require computation only on half of their components
  if(nfast%2==0)
    nfast_h = nfast/2+1;
  else
    nfast_h = (nfast+1)/2;

  if (allflag == 0) {
    reshape_preflag = 0;
    fast_ilo = in_ilo;
    fast_ihi = in_ihi;
    fast_jlo = in_jlo;
    fast_jhi = in_jhi;
    fast_klo = in_klo;
    fast_khi = in_khi;
  } else {
    reshape_preflag = 1;
    fast_ilo = 0;
    fast_ihi = nfast - 1;
    fast_jlo = ipfast2*nmid/npfast2;
    fast_jhi = (ipfast2+1)*nmid/npfast2 - 1;
    fast_klo = ipfast3*nslow/npfast3;
    fast_khi = (ipfast3+1)*nslow/npfast3 - 1;
  }

  insize = (in_ihi-in_ilo+1) * (in_jhi-in_jlo+1) * (in_khi-in_klo+1);
  fastsize = (fast_ihi-fast_ilo+1) * (fast_jhi-fast_jlo+1) * (fast_khi-fast_klo+1);
  fftsize = std::max(insize, fastsize);

  fft_fast = new FFT1d;

  fft_fast->length = nfast;
  fft_fast->n = (fast_jhi-fast_jlo+1) * (fast_khi-fast_klo+1);
  fft_fast->total = fft_fast->n * fft_fast->length;


// Create a class of processor for multi-dimensions FFT
  procfactors(nfast_h, 1, nslow,
              npmid1, npmid2, npmid3, ipmid1, ipmid2, ipmid3);
  procfactors(nfast_h,nmid, 1,
              npslow1, npslow2, npslow3, ipslow1, ipslow2, ipslow3);

// Reshape from fast to mid
  mid_ilo = ipmid1*nfast_h/npmid1;
  mid_ihi = (ipmid1+1)*nfast_h/npmid1 - 1;
  mid_jlo = 0;
  mid_jhi = nmid - 1;
  mid_klo = ipmid3*nslow/npmid3;
  mid_khi = (ipmid3+1)*nslow/npmid3 - 1;

//Reshape from mid to slow
  slow_ilo = ipslow1*nfast_h/npslow1;
  slow_ihi = (ipslow1+1)*nfast_h/npslow1 - 1;
  slow_jlo = ipslow2*nmid/npslow2;
  slow_jhi = (ipslow2+1)*nmid/npslow2 - 1;
  slow_klo = 0;
  slow_khi = nslow - 1;

// For use if half of the output need to be reconstructed
 procfactors(nfast, nmid, 1,
             npslow1_r2c, npslow2_r2c, npslow3_r2c, ipslow1_r2c, ipslow2_r2c, ipslow3_r2c);

  slow_ilo_r2c = ipslow1_r2c*nfast/npslow1_r2c;
  slow_ihi_r2c = (ipslow1_r2c+1)*nfast/npslow1_r2c - 1;
  slow_jlo_r2c = ipslow2_r2c*nmid/npslow2_r2c;
  slow_jhi_r2c = (ipslow2_r2c+1)*nmid/npslow2_r2c - 1;
  slow_klo_r2c = 0;
  slow_khi_r2c = nslow - 1;

// Check if reshape is needed from 3rd-direction (half-size brick) to original brick
  // if (out_ilo == slow_ilo_r2c && out_ihi == slow_ihi_r2c &&
  //     out_jlo == slow_jlo_r2c && out_jhi == slow_jhi_r2c &&
  //     out_klo == slow_klo_r2c && out_khi == slow_khi_r2c) flag = 0;
  // else flag = 1;
  // MPI_Allreduce(&flag, &allflag, 1, MPI_INT, MPI_MAX, world);
  //

  // if (allflag == 0){
  //    reshape_final_grid = 0;
  //    reshape_postflag = 0;
  // }
  // else {
    // reshape_final_grid = 1;
    // reshape_postflag = 1;
  // }


  // Note that if reshape_postflag = 1, the buffers will have a larger size than needed for the r2c transform
  sendsize = recvsize = 0;
  reshape_r2c_create(sendsize, recvsize);


  // insize/outsize = # of FFT data points in initial/final layout
  // fastsize/midsize/slowsize = # of data points in fast/mid/slow layout
  // maxsize = max of all these sizes, returned to caller

  outsize = (out_ihi-out_ilo+1) * (out_jhi-out_jlo+1) *
      (out_khi-out_klo+1);
  midsize = (mid_ihi-mid_ilo+1) * (mid_jhi-mid_jlo+1) *
    (mid_khi-mid_klo+1);
  slowsize = (slow_ihi-slow_ilo+1) * (slow_jhi-slow_jlo+1) *
    (slow_khi-slow_klo+1);

  fftsize = std::max(fftsize, outsize);
  fftsize = std::max(insize, midsize);
  fftsize = std::max(fftsize, slowsize);

  fft_mid = new FFT1d;
  fft_slow = new FFT1d;

  fft_mid->length = nmid;
  fft_mid->n = (mid_ihi-mid_ilo+1) * (mid_khi-mid_klo+1);
  fft_mid->total = fft_mid->n * fft_mid->length;

  fft_slow->length = nslow;
  fft_slow->n = (slow_ihi-slow_ilo+1) * (slow_jhi-slow_jlo+1);
  fft_slow->total = fft_slow->n * fft_slow->length;

  setup_ffts_r2c();

  norm = 1.0/((bigint) nfast * nmid*nslow);
  normnum = outsize;

  // allocate sendbuf, recvbuf arrays to max sizes needed by any reshape
  if (memoryflag) {
    setup_memory_flag = 1;
    if (sendsize) {
      sendbuf = (U *) memory->smalloc(sendsize*sizeof(U), mem_type);
      if (!sendbuf) error->one("Could not allocate sendbuf array");
    }
    if (recvsize) {
      recvbuf = (U *) memory->smalloc(recvsize*sizeof(U), mem_type);
      if (!recvbuf) error->one("Could not allocate recvbuf array");
    }
  }

  // return buffer sizes to caller

  user_fftsize  = fftsize;
  user_sendsize = sendsize;
  user_recvsize = recvsize;

  // set memusage for FFT and Reshape memory

  memusage = 0;

  if (memoryflag) {
    memusage += (int64_t) sendsize * sizeof(U);
    memusage += (int64_t) recvsize * sizeof(U);
  }

  memusage += reshape_memory();

}

template
void FFT3d<double>::setup_r2c(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                              int &user_fftsize, int &user_sendsize, int &user_recvsize);
template
void FFT3d<float>::setup_r2c(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi,
                              int &user_fftsize, int &user_sendsize, int &user_recvsize);





/* ----------------------------------------------------------------------
  Deallocate memory allocated by setup()
------------------------------------------------------------------------- */

template <class U>
void FFT3d<U>::deallocate_setup()
{
  setupflag = 0;

  deallocate_reshape(reshape_prefast);
  deallocate_reshape(reshape_fastmid);
  deallocate_reshape(reshape_midslow);
  deallocate_reshape(reshape_postslow);

  deallocate_reshape(reshape_preslow);
  deallocate_reshape(reshape_slowmid);
  deallocate_reshape(reshape_midfast);
  deallocate_reshape(reshape_postfast);

  deallocate_ffts();
  delete fft_fast;
  delete fft_mid;
  delete fft_slow;

  reshape_prefast = reshape_fastmid = reshape_midslow = reshape_postslow = NULL;
  reshape_preslow = reshape_slowmid = reshape_midfast = reshape_postfast = NULL;
  fft_fast = fft_mid = fft_slow = NULL;
}

template
void FFT3d<double>::deallocate_setup();
template
void FFT3d<float>::deallocate_setup();



/* ----------------------------------------------------------------------
  Deallocate memory allocated by setup_r2c()
------------------------------------------------------------------------- */

template <class U>
void FFT3d<U>::deallocate_setup_r2c()
{
  setupflag_r2c = 0;

  deallocate_reshape(reshape_prefast);
  deallocate_reshape(reshape_fastmid);
  deallocate_reshape(reshape_midslow);
  deallocate_reshape(reshape_postslow);

  reshape_prefast = reshape_fastmid = reshape_midslow = reshape_postslow = NULL;

  deallocate_ffts_r2c();
  delete fft_fast;
  delete fft_mid;
  delete fft_slow;

  fft_fast = fft_mid = fft_slow = NULL;
}

template
void FFT3d<double>::deallocate_setup_r2c();
template
void FFT3d<float>::deallocate_setup_r2c();


/* ----------------------------------------------------------------------
   pass in user memory for Reshape send/recv operations
   user_sendbuf = send buffer of length user_sendsize
   user_recvbuf = send buffer of length user_recvsize
------------------------------------------------------------------------- */

template <class U>
template <class T>
void FFT3d<U>::setup_memory(T *user_sendbuf, T *user_recvbuf)
{
  if (!setupflag) error->all("Cannot setup FFT memory before setup");
  setup_memory_flag = 1;
  // sendbuf = (T *)user_sendbuf;
  // recvbuf = (T *)user_recvbuf;
}

template
void FFT3d<double>::setup_memory(double *user_sendbuf, double *user_recvbuf);
template
void FFT3d<float>::setup_memory(float *user_sendbuf, float *user_recvbuf);


/* ----------------------------------------------------------------------
   deallocate memory allocated internally for send/recv
   only called if allocated internally
------------------------------------------------------------------------- */


template <class U>
void FFT3d<U>::deallocate_setup_memory()
{
  setup_memory_flag = 0;
  memory->sfree(sendbuf, mem_type);
  memory->sfree(recvbuf, mem_type);
  sendbuf = recvbuf = NULL;
}






/**
 * Perform a 3D C2C FFT
 * @param in Address of input data on this proc
 * @param out Address of output data on this proc (can be same as in)
 * @param flag  -1 for forward FFT, 1 for inverse FFT
 */
template <class U>
template <class T>
void FFT3d<U>::compute(T *in, T *out, int flag)
{
  int  thread_id = 0;
  char func_name[80], func_message[80];
  T fft_norm;

  if (!setupflag) error->all("Cannot compute FFT before setup");
  if (!setup_memory_flag) error->all("Cannot compute FFT before setup_memory");

  T *data = out;

  double t;
  t = MPI_Wtime();

  if (flag == 1 || inout_layout_same) {

    if (reshape_prefast) {
      snprintf(func_name, sizeof(func_name), "reshape_prefast");
      snprintf(func_message, sizeof(func_message), "reshape_prefast");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(in, out, reshape_prefast);
      trace_cpu_end( thread_id);
    }
    else if (in != out) { // TODO: add cuda copy
      snprintf(func_name, sizeof(func_name), "memcpy");
      snprintf(func_message, sizeof(func_message), "in != out");
      trace_cpu_start( thread_id, func_name, func_message );
      memcpy(out, in, insize*sizeof(T));
      trace_cpu_end( thread_id);
    }

    if (reshapeonly) {
      if (reshape_fastmid) {
        snprintf(func_name, sizeof(func_name), "reshape_fastmid");
        snprintf(func_message, sizeof(func_message), "RESHAPEONLY:reshape_fastmid");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_fastmid);
	trace_cpu_end( thread_id);
      }
      if (reshape_midslow) {
        snprintf(func_name, sizeof(func_name), "reshape_midslow");
        snprintf(func_message, sizeof(func_message), "RESHAPEONLY:reshape_midslow");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_midslow);
	trace_cpu_end( thread_id);
      }
    } else {
      snprintf(func_name, sizeof(func_name), "compute_fast");
      snprintf(func_message, sizeof(func_message), "compute_fast");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_fast);
      }
      trace_cpu_end( thread_id);
      if (reshape_fastmid) {
        snprintf(func_name, sizeof(func_name), "reshape_fastmid");
        snprintf(func_message, sizeof(func_message), "reshape_fastmid");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_fastmid);
      	trace_cpu_end( thread_id);
      }
      snprintf(func_name, sizeof(func_name), "compute_mid");
      snprintf(func_message, sizeof(func_message), "compute_mid");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_mid);
      }
      trace_cpu_end( thread_id);
      if (reshape_midslow) {
        snprintf(func_name, sizeof(func_name), "reshape_midslow");
        snprintf(func_message, sizeof(func_message), "reshape_midslow");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_midslow);
	      trace_cpu_end( thread_id);
      }
      snprintf(func_name, sizeof(func_name), "compute_slow");
      snprintf(func_message, sizeof(func_message), "compute_slow");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_slow);
      }
      trace_cpu_end( thread_id);
    }

    if (reshape_postslow) {
      snprintf(func_name, sizeof(func_name), "reshape_postslow");
      snprintf(func_message, sizeof(func_message), "reshape_postslow");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_postslow);
      trace_cpu_end( thread_id);
    }

    if (flag == 1 && scaled && !reshapeonly) {
      snprintf(func_name, sizeof(func_name), "scale_fft");
      snprintf(func_message, sizeof(func_message), "scale_fft");
      trace_cpu_start( thread_id, func_name, func_message );
      heffte::add_trace name("scale");
      scale_ffts(fft_norm, data);
      trace_cpu_end( thread_id);
    }

  } else {

    if (reshape_preslow) {
      snprintf(func_name, sizeof(func_name), "reshape_preslow");
      snprintf(func_message, sizeof(func_message), "reshape_preslow");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(in, out, reshape_preslow);
      trace_cpu_end( thread_id);
    }
    else if (in != out) {
    snprintf(func_name, sizeof(func_name), "memcpy");
    snprintf(func_message, sizeof(func_message), "in != out");
    trace_cpu_start( thread_id, func_name, func_message );
    memcpy(out, in, outsize*sizeof(T));
    trace_cpu_end( thread_id);
    }

    if (reshapeonly) {
      if (reshape_slowmid) {
        snprintf(func_name, sizeof(func_name), "reshape_slowmid");
        snprintf(func_message, sizeof(func_message), "reshape_slowmid");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_slowmid);
	      trace_cpu_end( thread_id);
      }
      if (reshape_midfast) {
        snprintf(func_name, sizeof(func_name), "reshape_midfast");
        snprintf(func_message, sizeof(func_message), "reshape_midfast");
        trace_cpu_start( thread_id, func_name, func_message );
        reshape(data, data, reshape_midfast);
	      trace_cpu_end( thread_id);
      }
    } else {
      snprintf(func_name, sizeof(func_name), "compute_slow");
      snprintf(func_message, sizeof(func_message), "compute_slow");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_slow);
      }
      trace_cpu_end( thread_id);
      if (reshape_slowmid) {
      snprintf(func_name, sizeof(func_name), "reshape_slowmid");
      snprintf(func_message, sizeof(func_message), "reshape_slowmid");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_slowmid);
      trace_cpu_end( thread_id);
      }
      snprintf(func_name, sizeof(func_name), "compute_mid");
      snprintf(func_message, sizeof(func_message), "compute_mid");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_mid);
      }
      trace_cpu_end( thread_id);
      if (reshape_midfast) {
      snprintf(func_name, sizeof(func_name), "reshape_midfast");
      snprintf(func_message, sizeof(func_message), "reshape_midfast");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_midfast);
      trace_cpu_end( thread_id);
      }
      snprintf(func_name, sizeof(func_name), "compute_fast");
      snprintf(func_message, sizeof(func_message), "compute_fast");
      trace_cpu_start( thread_id, func_name, func_message );
      { heffte::add_trace name("fft1d");
      perform_ffts(data,flag,fft_fast);
      }
      trace_cpu_end( thread_id);
    }

    if (reshape_postfast) {
    snprintf(func_name, sizeof(func_name), "reshape_postfast");
    snprintf(func_message, sizeof(func_message), "reshape_postfast");
    trace_cpu_start( thread_id, func_name, func_message );
    reshape(in, out, reshape_postfast);
    trace_cpu_end( thread_id);
    }
  }

  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[0] += MPI_Wtime() - t;
  #endif
}

template
void FFT3d<double>::compute(double *in, double *out, int flag);
template
void FFT3d<float>::compute(float *in, float *out, int flag);


























/**
 * Perform a 3D R2C FFT
 * @param in Address of input data on this proc
 * @param out Address of output data on this proc (can be same as in)
 */
template <class U>
template <class T>
void FFT3d<U>::compute_r2c(T *in, T *out)
{
  int  thread_id = 0;
  char func_name[80], func_message[80];
  T fft_norm;
  int flag = -1; // Forward FFT

  if (!setupflag_r2c) error->all("Cannot compute FFT before setup");
  if (!setup_memory_flag) error->all("Cannot compute FFT before setup_memory");

  T *data = out;

  double t;
  t = MPI_Wtime();

    if (reshape_prefast) {
      snprintf(func_name, sizeof(func_name), "reshape_prefast");
      snprintf(func_message, sizeof(func_message), "reshape_prefast");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(in, out, reshape_prefast);
      trace_cpu_end( thread_id);
    }

    snprintf(func_name, sizeof(func_name), "compute_r2c_fast");
    snprintf(func_message, sizeof(func_message), "compute_r2c_fast");
    trace_cpu_start( thread_id, func_name, func_message );
    { heffte::add_trace name("fft1d");
    perform_ffts_r2c(data, data,fft_fast);
    }
    trace_cpu_end( thread_id);

    if (reshape_fastmid) {
      snprintf(func_name, sizeof(func_name), "reshape_fastmid");
      snprintf(func_message, sizeof(func_message), "reshape_fastmid");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_fastmid);
    	trace_cpu_end( thread_id);
    }

    snprintf(func_name, sizeof(func_name), "compute_mid");
    snprintf(func_message, sizeof(func_message), "compute_mid");
    trace_cpu_start( thread_id, func_name, func_message );
    { heffte::add_trace name("fft1d");
    perform_ffts(data,flag,fft_mid);
    }
    trace_cpu_end( thread_id);

    if (reshape_midslow) {
      snprintf(func_name, sizeof(func_name), "reshape_midslow");
      snprintf(func_message, sizeof(func_message), "reshape_midslow");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_midslow);
      trace_cpu_end( thread_id);
    }

    snprintf(func_name, sizeof(func_name), "compute_slow");
    snprintf(func_message, sizeof(func_message), "compute_slow");
    trace_cpu_start( thread_id, func_name, func_message );
    { heffte::add_trace name("fft1d");
    perform_ffts(data,flag,fft_slow);
    }
    trace_cpu_end( thread_id);

    if (reshape_postslow) {
      snprintf(func_name, sizeof(func_name), "reshape_postslow");
      snprintf(func_message, sizeof(func_message), "reshape_postslow");
      trace_cpu_start( thread_id, func_name, func_message );
      reshape(data, data, reshape_postslow);
      trace_cpu_end( thread_id);
    }

  // if (scaled && !reshapeonly) {
  //   snprintf(func_name, sizeof(func_name), "scale_fft");
  //   snprintf(func_message, sizeof(func_message), "scale_fft");
  //   trace_cpu_start( thread_id, func_name, func_message );
  //   scale_ffts(fft_norm, data);
  //   trace_cpu_end( thread_id);
  // }

  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[0] += MPI_Wtime() - t;
  #endif

}

template
void FFT3d<double>::compute_r2c(double *in, double *out);
template
void FFT3d<float>::compute_r2c(float *in, float *out);




























/**
 * Perform just the 1d FFTs needed by a 3d FFT, no data movement
 * @param in starting address of input data on this proc, all set to 0.0
 * @param flag  1 for forward FFT, -1 for inverse FFT
 */
 template <class U>
 template <class T>
void FFT3d<U>::only_1d_ffts(T *in, int flag)
{
  if (!setupflag) error->all("Cannot compute 1d FFTs before setup");

  heffte::add_trace name("fft1d x3");
  perform_ffts(in,flag,fft_fast);
  perform_ffts(in,flag,fft_mid);
  perform_ffts(in,flag,fft_slow);
}

template
void FFT3d<double>::only_1d_ffts(double *in, int flag);
template
void FFT3d<float>::only_1d_ffts(float *in, int flag);


/**
 * Perform all the reshapes in a 3d FFT, but no 1d FFTs
 * @param in Address of input data on this proc
 * @param out address of output data on this proc (can be same as in)
 * @param flag  1 for forward FFT, -1 for inverse FFT
 */
 template <class U>
 template <class T>
void FFT3d<U>::only_reshapes(T *in, T *out, int flag)
{
  if (!setupflag) error->all("Cannot perform FFT reshape before setup");
  if (!setup_memory_flag)
    error->all("Cannot perform FFT reshape before setup_memory");

  T *data = out;

  if (flag == 1 || inout_layout_same) {

    if (reshape_prefast) reshape(in, out, reshape_prefast);
    else if (in != out) memcpy(out, in, insize*sizeof(T));

    if (reshape_fastmid) reshape(data, data, reshape_fastmid);
    if (reshape_midslow) reshape(data, data, reshape_midslow);

    if (reshape_postslow) reshape(data, data, reshape_postslow);

  } else {

    if (reshape_preslow) reshape(in, out, reshape_preslow);
    else if (in != out) memcpy(out, in, outsize*sizeof(T));

    if (reshape_slowmid) reshape(data, data, reshape_slowmid);
    if (reshape_midfast) reshape(data, data, reshape_midfast);

    if (reshape_postfast) reshape(data, data, reshape_postfast);
  }
}

template
void FFT3d<double>::only_reshapes(double *in, double *out, int flag);
template
void FFT3d<float>::only_reshapes(float *in, float *out, int flag);

/**
 * Perform just a single reshape operation
 * @param in Address of input data on this proc
 * @param out address of output data on this proc (can be same as in)
 * @param flag  1 for forward FFT, -1 for inverse FFT
 * @param which specify which reshape to perform = 1,2,3,4
 */
 template <class U>
 template <class T>
void FFT3d<U>::only_one_reshape(T *in, T *out, int flag, int which)
{
  if (!setupflag) error->all("Cannot perform an FFT reshape before setup");
  if (!setup_memory_flag)
    error->all("Cannot perform an FFT reshape before setup_memory");

  if (flag == 1 || inout_layout_same) {
    if (which == 1) {
      if (reshape_prefast) reshape(in, out, reshape_prefast);
      else if (in != out) memcpy(out, in, insize*sizeof(T));
    } else if (which == 2) {
      if (reshape_fastmid) reshape(in, out, reshape_fastmid);
    } else if (which == 3) {
      if (reshape_midslow) reshape(in, out, reshape_midslow);
    } else if (which == 4) {
      if (reshape_postslow) reshape(in, out, reshape_postslow);
    }

  } else {
    if (which == 4) {
      if (reshape_preslow) reshape(in, out, reshape_preslow);
      else if (in != out) memcpy(out, in, outsize*sizeof(T));
    } else if (which == 3) {
      if (reshape_slowmid) reshape(in, out, reshape_slowmid);
    } else if (which == 2) {
      if (reshape_midfast) reshape(in, out, reshape_midfast);
    } else if (which == 1) {
      if (reshape_postfast) reshape(in, out, reshape_postfast);
    }
  }
}

template
void FFT3d<double>::only_one_reshape(double *in, double *out, int flag, int which);
template
void FFT3d<float>::only_one_reshape(float *in, float *out, int flag, int which);


/**
 * Perform a 3d reshape of data
 * @param in Address of input data on this proc
 * @param out address of output data on this proc (can be same as in)
 * @param plan  Plan for reshape
 */
template <class U>
template <class T>
void FFT3d<U>::reshape(T *in, T *out, Reshape *plan)
{
  plan->reshape3d->reshape(in, out, sendbuf, recvbuf);
  if (plan->reshape3d_extra)
    plan->reshape3d_extra->reshape(in, out,sendbuf,recvbuf);
}

template
void FFT3d<double>::reshape(double *in, double *out, Reshape *plan);
template
void FFT3d<float>::reshape(float *in, float *out, Reshape *plan);


/* ----------------------------------------------------------------------
   dellocate a Reshape and its contents
------------------------------------------------------------------------- */

template <class U>
void FFT3d<U>::deallocate_reshape(Reshape *reshape)
{
  if (reshape == NULL) return;
  delete reshape->reshape3d;
  delete reshape->reshape3d_extra;
  delete reshape;
}

template
void FFT3d<double>::deallocate_reshape(Reshape *reshape);
template
void FFT3d<float>::deallocate_reshape(Reshape *reshape);

/**
 * Create plans for reshaping at all stages of forward 3D FFT computation
 * @param sendsize Size of sending buffer for inter-process communication
 * @param recvsize Size of receiving buffer for inter-process communication
 */
template <class U>
void FFT3d<U>::reshape_forward_create(int &sendsize, int &recvsize)
{
  int ssize, rsize;

  // reshape uses I=fast, J=mid, K=slow, b/c current permute=0
  if (reshape_preflag) {
    reshape_prefast = new Reshape;
    reshape_prefast->reshape3d = new Reshape3d<U>(world);
    reshape_prefast->reshape3d->memory_type = mem_type;
    reshape_prefast->reshape3d->collective = collective_bp;
    reshape_prefast->reshape3d->packflag = packflag;
    reshape_prefast->reshape3d->
      setup(in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,
            fast_ilo, fast_ihi,fast_jlo, fast_jhi,fast_klo, fast_khi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_prefast->reshape3d_extra = NULL;
  }

  // if exchange = 0, reshape direct from pencil to pencil
  // if exchange = 1, two reshapes from pencil to brick, then brick to pencil
  // reshape uses I=fast, J=mid, K=slow, b/c current permute=0

  reshape_fastmid = new Reshape;
  if (exchange == 0) {
    reshape_fastmid->reshape3d = new Reshape3d<U>(world);
    reshape_fastmid->reshape3d->memory_type = mem_type;
    reshape_fastmid->reshape3d->collective = collective_pp;
    reshape_fastmid->reshape3d->packflag = packflag;
    reshape_fastmid->reshape3d->
      setup(fast_ilo, fast_ihi,fast_jlo, fast_jhi,fast_klo, fast_khi,
            mid_ilo, mid_ihi,mid_jlo, mid_jhi,mid_klo, mid_khi,
            2, 1,0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_fastmid->reshape3d_extra = NULL;

  } else {
    reshape_fastmid->reshape3d = new Reshape3d<U>(world);
    reshape_fastmid->reshape3d->memory_type = mem_type;
    reshape_fastmid->reshape3d->collective = collective_bp;
    reshape_fastmid->reshape3d->packflag = packflag;
    reshape_fastmid->reshape3d->
      setup(fast_ilo, fast_ihi,fast_jlo, fast_jhi,fast_klo, fast_khi,
            brick_ilo, brick_ihi,brick_jlo, brick_jhi,brick_klo, brick_khi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_fastmid->reshape3d_extra = new Reshape3d<U>(world);
    reshape_fastmid->reshape3d->memory_type = mem_type;
    reshape_fastmid->reshape3d_extra->collective = collective_bp;
    reshape_fastmid->reshape3d_extra->packflag = packflag;
    reshape_fastmid->reshape3d_extra->
      setup(brick_ilo, brick_ihi,brick_jlo, brick_jhi,brick_klo, brick_khi,
            mid_ilo, mid_ihi,mid_jlo, mid_jhi,mid_klo, mid_khi,
            2, 1,0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
  }

  // if exchange = 0, reshape direct from pencil to pencil
  // if exchange = 1, two reshapes from pencil to brick, then brick to pencil
  // reshape uses J=fast, K=mid, I=slow, b/c current permute=1

  reshape_midslow = new Reshape;
  if (exchange == 0) {
    reshape_midslow->reshape3d = new Reshape3d<U>(world);
    reshape_midslow->reshape3d->memory_type = mem_type;
    reshape_midslow->reshape3d->collective = collective_pp;
    reshape_midslow->reshape3d->packflag = packflag;
    reshape_midslow->reshape3d->
      setup(mid_jlo, mid_jhi,mid_klo, mid_khi,mid_ilo, mid_ihi,
            slow_jlo, slow_jhi, slow_klo, slow_khi, slow_ilo, slow_ihi,
            2, 1,0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_midslow->reshape3d_extra = NULL;

  } else {
    reshape_midslow->reshape3d = new Reshape3d<U>(world);
    reshape_midslow->reshape3d->memory_type = mem_type;
    reshape_midslow->reshape3d->collective = collective_bp;
    reshape_midslow->reshape3d->packflag = packflag;
    reshape_midslow->reshape3d->
      setup(mid_jlo, mid_jhi,mid_klo, mid_khi,mid_ilo, mid_ihi,
            brick_jlo, brick_jhi,brick_klo, brick_khi,brick_ilo, brick_ihi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_midslow->reshape3d_extra = new Reshape3d<U>(world);
    reshape_midslow->reshape3d->memory_type = mem_type;
    reshape_midslow->reshape3d_extra->collective = collective_bp;
    reshape_midslow->reshape3d_extra->packflag = packflag;
    reshape_midslow->reshape3d_extra->
      setup(brick_jlo, brick_jhi,brick_klo, brick_khi,brick_ilo, brick_ihi,
            slow_jlo, slow_jhi, slow_klo, slow_khi, slow_ilo, slow_ihi,
            2, 1,0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
  }

  // reshape uses K=fast, I=mid, J=slow, b/c current permute=2
  // newpermute is from current permute=2 to desired permute=user_permute

  if (reshape_postflag) {
    reshape_postslow = new Reshape;
    int newpermute;
    if (permute == 0) newpermute = 1;
    if (permute == 1) newpermute = 2;
    if (permute == 2) newpermute = 0;
    reshape_postslow->reshape3d = new Reshape3d<U>(world);
    reshape_postslow->reshape3d->memory_type = mem_type;
    reshape_postslow->reshape3d->collective = collective_bp;
    reshape_postslow->reshape3d->packflag = packflag;
    reshape_postslow->reshape3d->
      setup(slow_klo, slow_khi, slow_ilo, slow_ihi, slow_jlo, slow_jhi,
            out_klo,out_khi,out_ilo,out_ihi,out_jlo,out_jhi,
            2, newpermute,0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_postslow->reshape3d_extra = NULL;
  }
}

template
void FFT3d<double>::reshape_forward_create(int &sendsize, int &recvsize);
template
void FFT3d<float>::reshape_forward_create(int &sendsize, int &recvsize);








template <class U>
void FFT3d<U>::reshape_r2c_create(int &sendsize, int &recvsize)
{
  int ssize, rsize;

  if (reshape_preflag) {
    reshape_prefast = new Reshape;
    reshape_prefast->reshape3d = new Reshape3d<U>(world);
    reshape_prefast->reshape3d->memory_type = mem_type;
    reshape_prefast->reshape3d->collective = collective_bp;
    reshape_prefast->reshape3d->packflag = packflag;
    reshape_prefast->reshape3d->
      setup(in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi,
            fast_ilo, fast_ihi, fast_jlo, fast_jhi, fast_klo, fast_khi,
            1, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_prefast->reshape3d_extra = NULL;
  }


  reshape_fastmid = new Reshape;
  reshape_fastmid->reshape3d = new Reshape3d<U>(world);
  reshape_fastmid->reshape3d->memory_type = mem_type;
  reshape_fastmid->reshape3d->collective = collective_pp;
  reshape_fastmid->reshape3d->packflag = packflag;
  reshape_fastmid->reshape3d->
    setup(fast_ilo, fast_ihi, fast_jlo, fast_jhi, fast_klo, fast_khi,
          mid_ilo, mid_ihi, mid_jlo, mid_jhi, mid_klo, mid_khi,
          2, 1, 0, ssize, rsize);
  sendsize = std::max(sendsize, ssize);
  recvsize = std::max(recvsize, rsize);
  reshape_fastmid->reshape3d_extra = NULL;


  reshape_midslow = new Reshape;
  reshape_midslow->reshape3d = new Reshape3d<U>(world);
  reshape_midslow->reshape3d->memory_type = mem_type;
  reshape_midslow->reshape3d->collective = collective_pp;
  reshape_midslow->reshape3d->packflag = packflag;
  reshape_midslow->reshape3d->
    setup(mid_jlo, mid_jhi, mid_klo, mid_khi, mid_ilo, mid_ihi,
          slow_jlo, slow_jhi, slow_klo, slow_khi, slow_ilo, slow_ihi,
          2, 1, 0, ssize, rsize);
  sendsize = std::max(sendsize, ssize);
  recvsize = std::max(recvsize, rsize);
  reshape_midslow->reshape3d_extra = NULL;


  if (reshape_postflag) {
    reshape_postslow = new Reshape;
    int newpermute;
    if (permute == 0) newpermute = 1;
    if (permute == 1) newpermute = 2;
    if (permute == 2) newpermute = 0;
    reshape_postslow->reshape3d = new Reshape3d<U>(world);
    reshape_postslow->reshape3d->memory_type = mem_type;
    reshape_postslow->reshape3d->collective = collective_bp;
    reshape_postslow->reshape3d->packflag = packflag;
    reshape_postslow->reshape3d->
      setup(slow_klo_r2c, slow_khi_r2c, slow_ilo_r2c, slow_ihi_r2c, slow_jlo_r2c, slow_jhi_r2c,
            out_klo, out_khi, out_ilo, out_ihi, out_jlo, out_jhi,
            2, newpermute, 1, ssize, rsize);
    reshape_postslow->reshape3d_extra = NULL;
  }
}

template
void FFT3d<double>::reshape_r2c_create(int &sendsize, int &recvsize);
template
void FFT3d<float>::reshape_r2c_create(int &sendsize, int &recvsize);































/**
 * Create plans for reshaping at all stages of backward 3D FFT computation
 * @param sendsize Size of sending buffer for inter-process communication
 * @param recvsize Size of receiving buffer for inter-process communication
 */
template <class U>
void FFT3d<U>::reshape_inverse_create(int &sendsize, int &recvsize)
{
  int ssize, rsize;

  // if current permute=0. reshape uses I=fast, J=mid, K=slow
  // if current permute=1, reshape uses J=fast, K=mid, I=slow
  // if current permute=2, reshape uses K=fast, I=mid, J=slow

  if (reshape_postflag) {
    reshape_preslow = new Reshape();
    if (permute == 0) {
      reshape_preslow->reshape3d = new Reshape3d<U>(world);
      reshape_preslow->reshape3d->memory_type = mem_type;
      reshape_preslow->reshape3d->collective = collective_bp;
      reshape_preslow->reshape3d->packflag = packflag;
      reshape_preslow->reshape3d->
        setup(out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi,
              slow_ilo, slow_ihi, slow_jlo, slow_jhi, slow_klo, slow_khi,
              2, 2, 0, ssize, rsize);
      sendsize = std::max(sendsize, ssize);
      recvsize = std::max(recvsize, rsize);
    } else if (permute == 1) {
      reshape_preslow->reshape3d = new Reshape3d<U>(world);
      reshape_preslow->reshape3d->memory_type = mem_type;
      reshape_preslow->reshape3d->collective = collective_bp;
      reshape_preslow->reshape3d->packflag = packflag;
      reshape_preslow->reshape3d->
        setup(out_jlo, out_jhi, out_klo, out_khi, out_ilo, out_ihi,
              slow_jlo, slow_jhi, slow_klo, slow_khi, slow_ilo, slow_ihi,
              2, 1, 0, ssize, rsize);
      sendsize = std::max(sendsize, ssize);
      recvsize = std::max(recvsize, rsize);
    } else if (permute == 2) {
      reshape_preslow->reshape3d = new Reshape3d<U>(world);
      reshape_preslow->reshape3d->memory_type = mem_type;
      reshape_preslow->reshape3d->collective = collective_bp;
      reshape_preslow->reshape3d->packflag = packflag;
      reshape_preslow->reshape3d->
        setup(out_klo, out_khi, out_ilo, out_ihi, out_jlo, out_jhi,
              slow_klo, slow_khi, slow_ilo, slow_ihi, slow_jlo, slow_jhi,
              2, 0, 0, ssize, rsize);
      sendsize = std::max(sendsize, ssize);
      recvsize = std::max(recvsize, rsize);
    }
    reshape_preslow->reshape3d_extra = NULL;
  }

  // if exchange = 0, reshape direct from pencil to pencil
  // if exchange = 1, two reshapes from pencil to brick, then brick to pencil
  // reshape uses K=fast, I=mid, J=slow, b/c current permute=2

  reshape_slowmid = new Reshape;
  if (exchange == 0) {
    reshape_slowmid->reshape3d = new Reshape3d<U>(world);
    reshape_slowmid->reshape3d->memory_type = mem_type;
    reshape_slowmid->reshape3d->collective = collective_pp;
    reshape_slowmid->reshape3d->packflag = packflag;
    reshape_slowmid->reshape3d->
      setup(slow_klo, slow_khi, slow_ilo, slow_ihi, slow_jlo, slow_jhi,
            mid_klo, mid_khi,mid_ilo, mid_ihi,mid_jlo, mid_jhi,
            2, 2, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_slowmid->reshape3d_extra = NULL;
  } else {
    reshape_slowmid->reshape3d = new Reshape3d<U>(world);
    reshape_slowmid->reshape3d->memory_type = mem_type;
    reshape_slowmid->reshape3d->collective = collective_bp;
    reshape_slowmid->reshape3d->packflag = packflag;
    reshape_slowmid->reshape3d->
      setup(slow_klo, slow_khi, slow_ilo, slow_ihi, slow_jlo, slow_jhi,
            brick_klo, brick_khi,brick_ilo, brick_ihi,brick_jlo, brick_jhi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_slowmid->reshape3d_extra = new Reshape3d<U>(world);
    reshape_slowmid->reshape3d->memory_type = mem_type;
    reshape_slowmid->reshape3d_extra->collective = collective_bp;
    reshape_slowmid->reshape3d_extra->packflag = packflag;
    reshape_slowmid->reshape3d_extra->
      setup(brick_klo, brick_khi,brick_ilo, brick_ihi,brick_jlo, brick_jhi,
            mid_klo, mid_khi,mid_ilo, mid_ihi,mid_jlo, mid_jhi,
            2, 2, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
  }

  // if exchange = 0, reshape direct from pencil to pencil
  // if exchange = 1, two reshapes from pencil to brick, then brick to pencil
  // reshape uses J=fast, K=mid, I=slow, b/c current permute=1

  reshape_midfast = new Reshape;
  if (exchange == 0) {
    reshape_midfast->reshape3d = new Reshape3d<U>(world);
    reshape_midfast->reshape3d->memory_type = mem_type;
    reshape_midfast->reshape3d->collective = collective_pp;
    reshape_midfast->reshape3d->packflag = packflag;
    reshape_midfast->reshape3d->
      setup(mid_jlo, mid_jhi,mid_klo, mid_khi,mid_ilo, mid_ihi,
            fast_jlo, fast_jhi,fast_klo, fast_khi,fast_ilo, fast_ihi,
            2, 2, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_midfast->reshape3d_extra = NULL;
  } else {
    reshape_midfast->reshape3d = new Reshape3d<U>(world);
    reshape_midfast->reshape3d->memory_type = mem_type;
    reshape_midfast->reshape3d->collective = collective_bp;
    reshape_midfast->reshape3d->packflag = packflag;
    reshape_midfast->reshape3d->
      setup(mid_jlo, mid_jhi,mid_klo, mid_khi,mid_ilo, mid_ihi,
            brick_jlo, brick_jhi,brick_klo, brick_khi,brick_ilo, brick_ihi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_midfast->reshape3d_extra = new Reshape3d<U>(world);
    reshape_midfast->reshape3d->memory_type = mem_type;
    reshape_midfast->reshape3d_extra->collective = collective_bp;
    reshape_midfast->reshape3d_extra->packflag = packflag;
    reshape_midfast->reshape3d_extra->
      setup(brick_jlo, brick_jhi,brick_klo, brick_khi,brick_ilo, brick_ihi,
            fast_jlo, fast_jhi,fast_klo, fast_khi,fast_ilo, fast_ihi,
            2, 2, 0, ssize, rsize);
  }

  // reshape uses I=fast, J=mid, K=slow, b/c current permute=0

  if (reshape_preflag) {
    reshape_postfast = new Reshape;
    reshape_postfast->reshape3d = new Reshape3d<U>(world);
    reshape_postfast->reshape3d->memory_type = mem_type;
    reshape_postfast->reshape3d->collective = collective_bp;
    reshape_postfast->reshape3d->packflag = packflag;
    reshape_postfast->reshape3d->
      setup(fast_ilo, fast_ihi,fast_jlo, fast_jhi,fast_klo, fast_khi,
            in_ilo,in_ihi,in_jlo,in_jhi,in_klo,in_khi,
            2, 0, 0, ssize, rsize);
    sendsize = std::max(sendsize, ssize);
    recvsize = std::max(recvsize, rsize);
    reshape_postfast->reshape3d_extra = NULL;
  }
}

template
void FFT3d<double>::reshape_inverse_create(int &sendsize, int &recvsize);
template
void FFT3d<float>::reshape_inverse_create(int &sendsize, int &recvsize);

/* ----------------------------------------------------------------------
   tally memory used by all Reshape3d instances
------------------------------------------------------------------------- */
template <class U>
int64_t FFT3d<U>::reshape_memory()
{
  int64_t memusage = 0;

  if (reshape_prefast) {
    memusage += reshape_prefast->reshape3d->memusage;
    if (reshape_prefast->reshape3d_extra)
      memusage += reshape_prefast->reshape3d_extra->memusage;
  }
  if (reshape_fastmid) {
    memusage += reshape_fastmid->reshape3d->memusage;
    if (reshape_fastmid->reshape3d_extra)
      memusage += reshape_fastmid->reshape3d_extra->memusage;
  }
  if (reshape_midslow) {
    memusage += reshape_midslow->reshape3d->memusage;
    if (reshape_midslow->reshape3d_extra)
      memusage += reshape_midslow->reshape3d_extra->memusage;
  }
  if (reshape_postslow) {
    memusage += reshape_postslow->reshape3d->memusage;
    if (reshape_postslow->reshape3d_extra)
      memusage += reshape_postslow->reshape3d_extra->memusage;
  }

  if (reshape_preslow) {
    memusage += reshape_preslow->reshape3d->memusage;
    if (reshape_preslow->reshape3d_extra)
      memusage += reshape_preslow->reshape3d_extra->memusage;
  }
  if (reshape_slowmid) {
    memusage += reshape_slowmid->reshape3d->memusage;
    if (reshape_slowmid->reshape3d_extra)
      memusage += reshape_slowmid->reshape3d_extra->memusage;
  }
  if (reshape_midfast) {
    memusage += reshape_midfast->reshape3d->memusage;
    if (reshape_midfast->reshape3d_extra)
      memusage += reshape_midfast->reshape3d_extra->memusage;
  }
  if (reshape_postfast) {
    memusage += reshape_postfast->reshape3d->memusage;
    if (reshape_postfast->reshape3d_extra)
      memusage += reshape_postfast->reshape3d_extra->memusage;
  }

  return memusage;
}
template
int64_t FFT3d<double>::reshape_memory();
template
int64_t FFT3d<float>::reshape_memory();

// -------------------------------------------------------------------
// -------------------------------------------------------------------
// FFT package specific code
// -------------------------------------------------------------------
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// Intel MKL FFTs
// -------------------------------------------------------------------

#if defined(FFT_MKL) || defined(FFT_MKL_OMP)


// ----------------------
// Complex to complex MKL
// ----------------------
// Plan definition

template <class U>
void FFT3d<U>::setup_ffts()
{
  fft1d = "MKL";

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_fast->handle, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG) nfast);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_fast->handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG) nfast);

  DftiSetValue(fft_fast->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_fast->total/nfast);
  DftiSetValue(fft_fast->handle, DFTI_PLACEMENT, DFTI_INPLACE);
  DftiSetValue(fft_fast->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nfast);
  DftiSetValue(fft_fast->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nfast);
  DftiCommitDescriptor(fft_fast->handle);

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_mid->handle, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG) nmid);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_mid->handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG) nmid);

  DftiSetValue(fft_mid->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_mid->total/nmid);
  DftiSetValue(fft_mid->handle, DFTI_PLACEMENT, DFTI_INPLACE);
  DftiSetValue(fft_mid->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nmid);
  DftiSetValue(fft_mid->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nmid);
  DftiCommitDescriptor(fft_mid->handle);

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_slow->handle, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG) nslow);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_slow->handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG) nslow);

  DftiSetValue(fft_slow->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_slow->total/nslow);
  DftiSetValue(fft_slow->handle, DFTI_PLACEMENT, DFTI_INPLACE);
  DftiSetValue(fft_slow->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nslow);
  DftiSetValue(fft_slow->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nslow);
  DftiCommitDescriptor(fft_slow->handle);
}


template <class U>
void FFT3d<U>::perform_ffts(U *data, int flag, FFT1d *plan)
{
  int  thread_id = 1;
  char func_name[80], func_message[80];

  using mkl_complex_type = typename mkl_traits<U>::mkl_complex_data;
  mkl_complex_type *mkl_data = (mkl_complex_type *) data;

  double t = MPI_Wtime();
  if (flag == -1) {
    snprintf(func_name, sizeof(func_name), "COMPUTE_FWD");
    snprintf(func_message, sizeof(func_message), "MKLDFTI_FWD%d",0);
    trace_cpu_start( thread_id, func_name, func_message );
    DftiComputeForward(plan->handle, mkl_data);
    trace_cpu_end( thread_id);
    }
  else {
    snprintf(func_name, sizeof(func_name), "COMPUTE_BWD");
    snprintf(func_message, sizeof(func_message), "MKLDFTI_BWD%d",0);
    trace_cpu_start( thread_id, func_name, func_message );
    DftiComputeBackward(plan->handle, mkl_data);
    trace_cpu_end( thread_id);
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1] += MPI_Wtime() - t;
  #endif
}

template <class U>
template <class T>
void FFT3d<U>::scale_ffts(T &fft_norm, T *data)
{
  int thread_id = 1;
  char func_name[80], func_message[80];
  snprintf(func_name, sizeof(func_name), "scale_ffts");
  snprintf(func_message, sizeof(func_message), "scale_fft_MKL");
  trace_cpu_start( thread_id, func_name, func_message );
  fft_norm = norm;
  T *data_ptr = (T *) data;
  double t = MPI_Wtime();
  for (int i = 0; i < normnum; i++) {
    *(data_ptr++) *= fft_norm;
    *(data_ptr++) *= fft_norm;
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[4] += MPI_Wtime() - t;
  #endif
  trace_cpu_end( thread_id);
}



// -------------------
// Real to complex MKL
// -------------------
// Plan definition

template <class U>
void FFT3d<U>::setup_ffts_r2c()
{
  fft1d = "MKL";

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_fast->handle, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG) nfast);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_fast->handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) nfast);

  DftiSetValue(fft_fast->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_fast->total/nfast);
  DftiSetValue(fft_fast->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(fft_fast->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nfast);
  DftiSetValue(fft_fast->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nfast);
  DftiCommitDescriptor(fft_fast->handle);

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_mid->handle, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG) nmid);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_mid->handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) nmid);

  DftiSetValue(fft_mid->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_mid->total/nmid);
  DftiSetValue(fft_mid->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(fft_mid->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nmid);
  DftiSetValue(fft_mid->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nmid);
  DftiCommitDescriptor(fft_mid->handle);

  if(sizeof(U)==4) DftiCreateDescriptor(&fft_slow->handle, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG) nslow);
  if(sizeof(U)==8) DftiCreateDescriptor(&fft_slow->handle, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) nslow);

  DftiSetValue(fft_slow->handle, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) fft_slow->total/nslow);
  DftiSetValue(fft_slow->handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(fft_slow->handle, DFTI_INPUT_DISTANCE, (MKL_LONG) nslow);
  DftiSetValue(fft_slow->handle, DFTI_OUTPUT_DISTANCE, (MKL_LONG) nslow);
  DftiCommitDescriptor(fft_slow->handle);
}




template <class U>
void FFT3d<U>::perform_ffts_r2c(U *data, U *data_out, FFT1d *plan)
{
  int  thread_id = 1;
  char func_name[80], func_message[80];

  using mkl_complex_type = typename mkl_traits<U>::mkl_complex_data;
  mkl_complex_type *mkl_data = (mkl_complex_type *) data_out;

  double t = MPI_Wtime();
  snprintf(func_name, sizeof(func_name), "COMPUTE_R2C");
  snprintf(func_message, sizeof(func_message), "MKLDFTI_FWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
  DftiComputeForward(plan->handle, data, mkl_data);
  trace_cpu_end( thread_id);
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1] += MPI_Wtime() - t;
  #endif
}





template <class U>
void FFT3d<U>::deallocate_ffts()
{
  DftiFreeDescriptor(&fft_fast->handle);
  DftiFreeDescriptor(&fft_mid->handle);
  DftiFreeDescriptor(&fft_slow->handle);
}


template <class U>
void FFT3d<U>::deallocate_ffts_r2c()
{
  DftiFreeDescriptor(&fft_fast->handle);
  DftiFreeDescriptor(&fft_mid->handle);
  DftiFreeDescriptor(&fft_slow->handle);
}




// -------------------------------------------------------------------
// FFTW2 FFTs
// -------------------------------------------------------------------

#elif defined(FFT_FFTW2)

template <class U>
void FFT3d<U>::setup_ffts()
{
  fft1d = "FFTW2";

  fft_fast->plan_forward =
    fftw_create_plan(nfast,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  fft_fast->plan_backward =
    fftw_create_plan(nfast,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);

  if (nmid == nfast) {
    fft_mid->plan_forward = fft_fast->plan_forward;
    fft_mid->plan_backward = fft_fast->plan_backward;
  } else {
    fft_mid->plan_forward =
      fftw_create_plan(nmid,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
    fft_mid->plan_backward =
      fftw_create_plan(nmid,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  }

  if (nslow == nfast) {
    fft_slow->plan_forward = fft_fast->plan_forward;
    fft_slow->plan_backward = fft_fast->plan_backward;
  } else if (nslow == nmid) {
    fft_slow->plan_forward = fft_mid->plan_forward;
    fft_slow->plan_backward = fft_mid->plan_backward;
  } else {
    fft_slow->plan_forward =
      fftw_create_plan(nslow,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
    fft_slow->plan_backward =
      fftw_create_plan(nslow,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_IN_PLACE);
  }
}

template <class U>
void FFT3d<U>::perform_ffts(U *data, int flag, FFT1d *plan)
{
  fftw_complex *data_fftw2 = (fftw_complex *) data;
  int  thread_id = 1;
  char func_name[80], func_message[80];
  double t;
  t = MPI_Wtime();
  if (flag == -1) {
    snprintf(func_name, sizeof(func_name), "COMPUTE_FWD");
    snprintf(func_message, sizeof(func_message), "FFTW2_FWD%d",0);
    trace_cpu_start( thread_id, func_name, func_message );
    fftw(plan->plan_forward,plan->n,data_fftw2, 1,plan->length,NULL,0,0);
    trace_cpu_end( thread_id);
    }
  else {
    snprintf(func_name, sizeof(func_name), "COMPUTE_BWD");
    snprintf(func_message, sizeof(func_message), "FFTW2_BWD%d",0);
    trace_cpu_start( thread_id, func_name, func_message );
    fftw(plan->plan_backward,plan->n,data_fftw2, 1,plan->length,NULL,0,0);
    trace_cpu_end( thread_id);
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1] += MPI_Wtime() - t;
  #endif
}

template <class U>
template <class T>
void FFT3d<U>::scale_ffts(T &fft_norm, T *data)
{
  int thread_id = 1;
  char func_name[80], func_message[80];
  snprintf(func_name, sizeof(func_name), "scale_ffts");
  snprintf(func_message, sizeof(func_message), "scale_fft_FFTW2");
  trace_cpu_start( thread_id, func_name, func_message );
  fft_norm = norm;
  for (int i = 0; i < normnum; i++) {
    data[i].re *= fft_norm;
    data[i].im *= fft_norm;
  }
}

template <class U>
void FFT3d<U>::deallocate_ffts()
{
  if (nslow != nfast && nslow != nmid) {
    fftw_destroy_plan(fft_slow->plan_forward);
    fftw_destroy_plan(fft_slow->plan_backward);
  }
  if (nmid != nfast) {
    fftw_destroy_plan(fft_mid->plan_forward);
    fftw_destroy_plan(fft_mid->plan_backward);
  }
  fftw_destroy_plan(fft_fast->plan_forward);
  fftw_destroy_plan(fft_fast->plan_backward);
}

// -------------------------------------------------------------------
// CUDA cuFFTs
// -------------------------------------------------------------------

#elif   defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

// ------------------------
// Complex to complex CUFFT
// ------------------------
// Plan definition

void cufft_plan_create_wrapper(cufftHandle &plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, int batch, double dummy)
{
  cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch);
}

void cufft_plan_create_wrapper(cufftHandle &plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, int batch, float dummy)
{
  cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
}

/**
 * Create plans for 1D FFTs for each direction
 */
template <class U>
void FFT3d<U>::setup_ffts()
{
  fft1d = "CUFFT";
  U dummy;

  int n = fft_fast->n;
  cufft_plan_create_wrapper((fft_fast->plan_unique), 1,  &nfast, &nfast, 1, nfast, &nfast, 1, nfast, n, dummy);
  heffte_check_cuda_error();

  n = fft_mid->n;
  cufft_plan_create_wrapper((fft_mid->plan_unique), 1, &nmid, &nmid, 1, nmid, &nmid, 1, nmid, n, dummy);
  heffte_check_cuda_error();

  n = fft_slow->n;
  cufft_plan_create_wrapper((fft_slow->plan_unique), 1, &nslow, &nslow, 1, nslow, &nslow, 1, nslow, n, dummy);
  heffte_check_cuda_error();

  cudaDeviceSynchronize();
}


// Execution

void cufft_execute_wrapper(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
  cufftExecC2C(plan, idata, odata, direction);
}
void cufft_execute_wrapper(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction)
{
  cufftExecZ2Z(plan, idata, odata, direction);
}

 /**
  * Perform 1D FFTs
  * @param data Address of input data this process
  * @param flag  1 for forward FFT, -1 for inverse FFT
  * @param plan Plan for 1D FFTs created by  \ref FFT3d::setup_ffts
  */
template <class U>
void FFT3d<U>::perform_ffts(U *data, int flag, FFT1d *plan)
{
  using complex_type = typename cufft_traits<U>::complex_type;
  complex_type *cufft_data = (complex_type *) data;

  int  thread_id = 1;
  char func_name[80], func_message[80];
  double t;
  t = MPI_Wtime();
  if (flag == -1) {
  snprintf(func_name, sizeof(func_name), "COMPUTE_FWD");
  snprintf(func_message, sizeof(func_message), "CUFFT_FWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
  cufft_execute_wrapper(plan->plan_unique, cufft_data, cufft_data, CUFFT_FORWARD);
  heffte_check_cuda_error();
  cudaDeviceSynchronize();
  heffte_check_cuda_error();
  trace_cpu_end( thread_id);
  }
  else {
  snprintf(func_name, sizeof(func_name), "COMPUTE_BWD");
  snprintf(func_message, sizeof(func_message), "CUFFT_BWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
  cufft_execute_wrapper(plan->plan_unique, cufft_data, cufft_data, CUFFT_INVERSE);
  heffte_check_cuda_error();
  cudaDeviceSynchronize();
  heffte_check_cuda_error();
  trace_cpu_end( thread_id);
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1]  += MPI_Wtime() - t;
  #endif
}


// ---------------------
// Real to complex CUFFT
// ---------------------

// Plan definition

void cufft_plan_create_r2c_wrapper(cufftHandle &plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, int batch, double dummy)
{
  cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, batch);
}

void cufft_plan_create_r2c_wrapper(cufftHandle &plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, int batch, float dummy)
{
  cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
}


template <class U>
void FFT3d<U>::setup_ffts_r2c()
{
  fft1d = "CUFFT";
  U dummy;

  int n = fft_fast->n;
  cufft_plan_create_r2c_wrapper((fft_fast->plan_unique), 1,  &nfast, &nfast, 1, nfast, &nfast, 1, nfast, n, dummy);
  heffte_check_cuda_error();

  n = fft_mid->n;
  cufft_plan_create_wrapper((fft_mid->plan_unique), 1, &nmid, &nmid, 1, nmid, &nmid, 1, nmid, n, dummy);
  heffte_check_cuda_error();

  n = fft_slow->n;
  cufft_plan_create_wrapper((fft_slow->plan_unique), 1, &nslow, &nslow, 1, nslow, &nslow, 1, nslow, n, dummy);
  heffte_check_cuda_error();

  cudaDeviceSynchronize();
}

// Execution

void cufft_execute_r2c_wrapper(cufftHandle plan, cufftReal *idata, cufftComplex *odata)
{
 cufftExecR2C(plan, idata, odata);
}
void cufft_execute_r2c_wrapper(cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata)
{
 cufftExecD2Z(plan, idata, odata);
}


template <class U>
void FFT3d<U>::perform_ffts_r2c(U *data, U *data_out, FFT1d *plan)
{
 using complex_type = typename cufft_traits<U>::complex_type;
 complex_type *cufft_data = (complex_type *) data_out;

 int  thread_id = 1;
 char func_name[80], func_message[80];
 double t;
 t = MPI_Wtime();
 snprintf(func_name, sizeof(func_name), "COMPUTE_R2C_CUFFT");
 snprintf(func_message, sizeof(func_message), "CUFFT_FWD%d",0);
 trace_cpu_start( thread_id, func_name, func_message );
 cufft_execute_r2c_wrapper(plan->plan_unique, data, cufft_data);
 heffte_check_cuda_error();
 cudaDeviceSynchronize();
 heffte_check_cuda_error();
 trace_cpu_end( thread_id);
 #if defined(HEFFTE_TIME_DETAILED)
  timing_array[1] += MPI_Wtime() - t;
 #endif
}


/**
 * Scale data after FFT computation
 * @param data Address of input data this process
 */
//
// extern "C" void scale_ffts_gpu(int n, double *data, double fnorm);

template <class U>
template <class T>
void FFT3d<U>::scale_ffts(T &fft_norm, T *data)
{
  int  thread_id = 1;
  char func_name[80], func_message[80];
  snprintf(func_name, sizeof(func_name), "scale_ffts");
  snprintf(func_message, sizeof(func_message), "scale_fft_CUFFT");
  trace_cpu_start( thread_id, func_name, func_message );
  fft_norm = norm;
  T *data_ptr = (T *) data;
  double t;
  t = MPI_Wtime();
  scale_ffts_gpu(2*normnum, data_ptr, norm);
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[4] += MPI_Wtime() - t;
  #endif
  trace_cpu_end( thread_id);
}



template <class U>
void FFT3d<U>::deallocate_ffts()
{
  cudaDeviceSynchronize();
  cufftDestroy(fft_fast->plan_unique);
  heffte_check_cuda_error();
  cufftDestroy(fft_mid->plan_unique);
  heffte_check_cuda_error();
  cufftDestroy(fft_slow->plan_unique);
  heffte_check_cuda_error();
}


template <class U>
void FFT3d<U>::deallocate_ffts_r2c()
{
  deallocate_ffts();
}



#else
// -------------------------------------------------------------------
// FFTW3 FFTs
// -------------------------------------------------------------------

// ------------------------
// Complex to complex FFTW3
// ------------------------
// Plan definition

fftw_plan fftw_plan_create_wrapper (int rank, const int *n, int howmany,
                                    fftw_complex *in, const int *inembed,
                                    int istride, int idist,
                                    fftw_complex *out, const int *onembed,
                                    int ostride, int odist,
                                    int sign, unsigned flags, double dummy)
{
  return fftw_plan_many_dft(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, sign,flags);
}

fftwf_plan fftw_plan_create_wrapper (int rank, const int *n, int howmany,
                                    fftw_complex *in, const int *inembed,
                                    int istride, int idist,
                                    fftw_complex *out, const int *onembed,
                                    int ostride, int odist,
                                    int sign, unsigned flags, float dummy)
{
  return fftwf_plan_many_dft(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, sign,flags);
}


template <class U>
void FFT3d<U>::setup_ffts()
{
  fft1d = "FFTW3";

  int n = fft_fast->n;
  U dummy;

  fft_fast->plan_forward  = fftw_plan_create_wrapper(1, &nfast, n, NULL, &nfast, 1, nfast, NULL, &nfast, 1, nfast,
                            FFTW_FORWARD,FFTW_ESTIMATE, dummy);
  fft_fast->plan_backward = fftw_plan_create_wrapper(1, &nfast, n, NULL, &nfast, 1, nfast, NULL, &nfast, 1, nfast,
                            FFTW_BACKWARD,FFTW_ESTIMATE, dummy);
  n = fft_mid->n;
  fft_mid->plan_forward   = fftw_plan_create_wrapper(1, &nmid, n, NULL, &nmid, 1, nmid, NULL, &nmid, 1, nmid,
                            FFTW_FORWARD,FFTW_ESTIMATE, dummy);
  fft_mid->plan_backward  = fftw_plan_create_wrapper(1, &nmid, n, NULL, &nmid, 1, nmid, NULL, &nmid, 1, nmid,
                            FFTW_BACKWARD,FFTW_ESTIMATE, dummy);
  n = fft_slow->n;
  fft_slow->plan_forward  = fftw_plan_create_wrapper(1, &nslow, n, NULL, &nslow, 1, nslow, NULL, &nslow, 1, nslow,
                            FFTW_FORWARD,FFTW_ESTIMATE, dummy);
  fft_slow->plan_backward = fftw_plan_create_wrapper(1, &nslow, n, NULL, &nslow, 1, nslow, NULL, &nslow, 1, nslow,
                            FFTW_BACKWARD,FFTW_ESTIMATE, dummy);
}

// Execution

void fftw_execute_wrapper(fftw_plan p, fftw_complex *in, fftw_complex *out)
{
   fftw_execute_dft(p, in, out);
}

void fftw_execute_wrapper(fftwf_plan p, fftwf_complex *in, fftwf_complex *out)
{
   fftwf_execute_dft(p, in, out);
}


template <class U>
void FFT3d<U>::perform_ffts(U *data, int flag, FFT1d *plan)
{
  using fftw_complex_type = typename fftw_traits<U>::fftw_complex_data;
  fftw_complex_type *fftw3_data = (fftw_complex_type *) data;

  int  thread_id = 1;
  char func_name[80], func_message[80];
  double t = MPI_Wtime();
  if (flag == -1) {
  snprintf(func_name, sizeof(func_name), "COMPUTE_FWD");
  snprintf(func_message, sizeof(func_message), "FFTW3_FWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
  fftw_execute_wrapper(plan->plan_forward, fftw3_data, fftw3_data);
  trace_cpu_end( thread_id);
  }
  else {
  snprintf(func_name, sizeof(func_name), "COMPUTE_BWD");
  snprintf(func_message, sizeof(func_message), "FFTW3_BWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
  fftw_execute_wrapper(plan->plan_backward, fftw3_data, fftw3_data);
  trace_cpu_end( thread_id);
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1] += MPI_Wtime() - t;
  #endif
}



// ---------------------
// Real to complex FFTW3
// ---------------------
// Plan definition

fftw_plan fftw_plan_create_r2c_wrapper (int rank, const int *n, int howmany,
                                    double *in, const int *inembed,
                                    int istride, int idist,
                                    fftw_complex *out, const int *onembed,
                                    int ostride, int odist,
                                    int sign, unsigned flags, double dummy)
{
  return fftw_plan_many_dft_r2c(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, flags);
}

fftwf_plan fftw_plan_create_r2c_wrapper (int rank, const int *n, int howmany,
                                    double *in, const int *inembed,
                                    int istride, int idist,
                                    fftw_complex *out, const int *onembed,
                                    int ostride, int odist,
                                    int sign, unsigned flags, float dummy)
{
  return fftwf_plan_many_dft_r2c(rank, n, howmany, NULL, inembed, istride, idist, NULL, onembed, ostride, odist, flags);
}



template <class U>
void FFT3d<U>::setup_ffts_r2c()
{
  fft1d = "FFTW3";

  U dummy;
  int n = fft_fast->n;

    fft_fast->plan_forward  = fftw_plan_create_r2c_wrapper(1, &nfast, n, NULL, &nfast, 1, nfast, NULL, &nfast, 1, nfast,
                            FFTW_FORWARD,FFTW_ESTIMATE, dummy);

    n = fft_mid->n;
    fft_mid->plan_forward   = fftw_plan_create_wrapper(1, &nmid, n, NULL, &nmid, 1, nmid, NULL, &nmid, 1, nmid,
                              FFTW_FORWARD,FFTW_ESTIMATE, dummy);

    n = fft_slow->n;
    fft_slow->plan_forward  = fftw_plan_create_wrapper(1, &nslow, n, NULL, &nslow, 1, nslow, NULL, &nslow, 1, nslow,
                              FFTW_FORWARD,FFTW_ESTIMATE, dummy);

}


// Execution

void fftw_execute_wrapper_r2c(fftw_plan p, double *in, fftw_complex *out)
{
   fftw_execute_dft_r2c(p, in, out);
}

void fftw_execute_wrapper_r2c(fftwf_plan p, float *in, fftwf_complex *out)
{
   fftwf_execute_dft_r2c(p, in, out);
}


template <class U>
void FFT3d<U>::perform_ffts_r2c(U *data, U *data_out, FFT1d *plan)
{
  using fftw_complex_type = typename fftw_traits<U>::fftw_complex_data;
  fftw_complex_type *fftw3_data = (fftw_complex_type *) data_out;

  int  thread_id = 1;
  char func_name[80], func_message[80];
  double t = MPI_Wtime();
  snprintf(func_name, sizeof(func_name), "COMPUTE_R2C_FFTW");
  snprintf(func_message, sizeof(func_message), "FFTW3_FWD%d",0);
  trace_cpu_start( thread_id, func_name, func_message );
    fftw_execute_wrapper_r2c(plan->plan_forward, data, fftw3_data);
  trace_cpu_end( thread_id);
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[1] += MPI_Wtime() - t;
  #endif
}






template <class U>
template <class T>
void FFT3d<U>::scale_ffts(T &fft_norm, T *data)
{
  int  thread_id = 1;
  char func_name[80], func_message[80];
  snprintf(func_name, sizeof(func_name), "scale_ffts");
  snprintf(func_message, sizeof(func_message), "scale_fft_FFTW3");
  trace_cpu_start( thread_id, func_name, func_message );
  fft_norm = norm;
  T *data_ptr = (T *) data;
  double t;
  t = MPI_Wtime();
  for (int i = 0; i < normnum; i++) {
    *(data_ptr++) *= fft_norm;
    *(data_ptr++) *= fft_norm;
  }
  #if defined(HEFFTE_TIME_DETAILED)
    timing_array[4] += MPI_Wtime() - t;
  #endif
  trace_cpu_end( thread_id);
}


// Template for deallocation of fftw3 variables
void fftw_deallocate_wrapper(fftw_plan p)
{
   fftw_destroy_plan(p);
}

void fftw_deallocate_wrapper(fftwf_plan p)
{
   fftwf_destroy_plan(p);
}

template <class U>
void FFT3d<U>::deallocate_ffts()
{
  fftw_deallocate_wrapper(fft_fast->plan_forward);
  fftw_deallocate_wrapper(fft_fast->plan_backward);
  fftw_deallocate_wrapper(fft_mid->plan_forward);
  fftw_deallocate_wrapper(fft_mid->plan_backward);
  fftw_deallocate_wrapper(fft_slow->plan_forward);
  fftw_deallocate_wrapper(fft_slow->plan_backward);
}


template <class U>
void FFT3d<U>::deallocate_ffts_r2c()
{
  fftw_deallocate_wrapper(fft_fast->plan_forward);
  fftw_deallocate_wrapper(fft_mid->plan_forward);
  fftw_deallocate_wrapper(fft_slow->plan_forward);
}

#endif

// -------------------------------------------------------------------
// -------------------------------------------------------------------
// end of FFT package specific code
// -------------------------------------------------------------------
// -------------------------------------------------------------------

/* ----------------------------------------------------------------------
   check if all prime factors of N are in list of prime factors
   return 1 if yes, 0 if no
------------------------------------------------------------------------- */
template <class U>
int FFT3d<U>::prime_factorable(int n)
{
  int i;

  while (n > 1) {
    for (i = 0; i < primes.size(); i++) {
      if (n % primes[i] == 0) {
        n /= primes[i];
        break;
      }
    }
    if (i == primes.size()) return 0;
  }

  return 1;
}

template
int FFT3d<double>::prime_factorable(int n);
template
int FFT3d<float>::prime_factorable(int n);

/* ----------------------------------------------------------------------
   computes factors of N up to sqrt(N)
   store ascending list in pre-allocated factors
   return nfactor
------------------------------------------------------------------------- */
template <class U>
void FFT3d<U>::factor(int n)
{
  int sqroot = (int) sqrt(n) + 1;
  if (sqroot*sqroot > n) sqroot--;

  factors.clear();
  for (int i = 1; i <= sqroot; i++) {
    if (n % i) continue;
    factors.push_back(i);
  }
}

template
void FFT3d<double>::factor(int n);
template
void FFT3d<float>::factor(int n);


/* ----------------------------------------------------------------------
   compute proc grid that is best match to FFT grid: Nx by Ny by Nz
   best = minimum surface area
   caller sets Nx or Ny or Nz = 1 if a 2d proc grid is desired
   else 3d if returned
   return npx, npy,npz = proc grid
   return ipx,ipy,ipz = my location in proc grid
------------------------------------------------------------------------- */
template <class U>
void FFT3d<U>::procfactors(int nx, int ny, int nz,
                       int &npx, int &npy, int &npz,
                       int &ipx, int &ipy, int &ipz)
{
  int i,j,jk,ifac, jfac, kfac;
  double newarea;

  int sqroot = (int) sqrt(nprocs) + 1;
  if (sqroot*sqroot > nprocs) sqroot--;

  double minarea = 2.0*nx*ny + 2.0*ny*nz + 2.0*nx*nz;

  // find 3d factorization of nprocs with min surface area for (Nx,Ny,Nz) grid
  // loop over all combinations of (ifac, jfac, kfac)
  // where ifac <= jfac and jfac <= kfac
  // then do surface-area test of all 6 permutations of (ifac, jfac, kfac)

  for (i = 0; i < factors.size(); i++) {
    ifac = factors[i];
    jk = nprocs/ifac;
    for (j = i; j < factors.size(); j++) {
      jfac = factors[j];
      kfac = jk/jfac;
      if (ifac*jfac*kfac != nprocs) continue;
      if (ifac > jfac || jfac > kfac) continue;

      newarea = surfarea(ifac, jfac, kfac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = ifac;
        npy = jfac;
        npz = kfac;
      }

      newarea = surfarea(ifac, kfac, jfac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = ifac;
        npy = kfac;
        npz = jfac;
      }

      newarea = surfarea(jfac,ifac, kfac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = jfac;
        npy = ifac;
        npz = kfac;
      }

      newarea = surfarea(jfac, kfac,ifac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = jfac;
        npy = kfac;
        npz = ifac;
      }

      newarea = surfarea(kfac,ifac, jfac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = kfac;
        npy = ifac;
        npz = jfac;
      }

      newarea = surfarea(kfac, jfac,ifac, nx, ny, nz);
      if (newarea < minarea) {
        minarea = newarea;
        npx = kfac;
        npy = jfac;
        npz = ifac;
      }
    }
  }

  // my location in 3d proc grid

  ipx = me % npx;
  ipy = (me/npx) % npy;
  ipz = me / (npx*npy);
}
template
void FFT3d<double>::procfactors(int nx, int ny, int nz,
                       int &npx, int &npy, int &npz,
                       int &ipx, int &ipy, int &ipz);
template
void FFT3d<float>::procfactors(int nx, int ny, int nz,
                       int &npx, int &npy, int &npz,
                       int &ipx, int &ipy, int &ipz);

/* ----------------------------------------------------------------------
   compute per-proc surface area for I,J,K proc grid and a Nx,Ny,Nz FFT grid
   if Nx or Ny or Nz = 1, force corresponding I,J,K to be 1, else return BIG
------------------------------------------------------------------------- */

template <class U>
double FFT3d<U>::surfarea(int i, int j, int k, int nx, int ny, int nz)
{
  if (nx == 1 && i != 1) return BIG;
  if (ny == 1 && j != 1) return BIG;
  if (nz == 1 && k != 1) return BIG;

  double dx = 1.0*nx/i;
  double dy = 1.0*ny/j;
  double dz = 1.0*nz/k;
  return dx*dy + dy*dz + dx*dz;
}
template
double FFT3d<double>::surfarea(int i, int j, int k, int nx, int ny, int nz);
template
double FFT3d<float>::surfarea(int i, int j, int k, int nx, int ny, int nz);


template
void FFT3d<double>::setup_ffts_r2c();
template
void FFT3d<float>::setup_ffts_r2c();

template
void FFT3d<double>::setup_ffts();
template
void FFT3d<float>::setup_ffts();

template
void FFT3d<double>::scale_ffts(double &fft_norm, double *data);
template
void FFT3d<float>::scale_ffts(float &fft_norm, float *data);

template
void FFT3d<double>::perform_ffts(double *data, int flag, FFT1d *plan);
template
void FFT3d<float>::perform_ffts(float *data, int flag, FFT1d *plan);


template
void FFT3d<double>::perform_ffts_r2c(double *data, double *data_out,  FFT1d *plan);
template
void FFT3d<float>::perform_ffts_r2c(float *data, float *data_out, FFT1d *plan);

template
void FFT3d<double>::deallocate_ffts();
template
void FFT3d<float>::deallocate_ffts();

#endif

#define heffte_instantiate_fft3d(some_backend) \
    template class fft3d<some_backend>; \
    template void fft3d<some_backend>::standard_transform<float>(float const[], std::complex<float>[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend>::standard_transform<double>(double const[], std::complex<double>[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \
    template void fft3d<some_backend>::standard_transform<float>(std::complex<float> const[], float[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend>::standard_transform<double>(std::complex<double> const[], double[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \
    template void fft3d<some_backend>::standard_transform<float>(std::complex<float> const[], std::complex<float>[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend>::standard_transform<double>(std::complex<double> const[], std::complex<double>[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \

namespace heffte {


template<typename backend_tag>
fft3d<backend_tag>::fft3d(logic_plan3d const &plan, int const this_mpi_rank, MPI_Comm const comm) :
    pinbox(plan.in_shape[0][this_mpi_rank]), poutbox(plan.out_shape[3][this_mpi_rank]),
    scale_factor(1.0 / static_cast<double>(plan.index_count))
{
    for(int i=0; i<4; i++){
        forward_shaper[i]    = make_reshape3d<backend_tag>(plan.in_shape[i], plan.out_shape[i], comm, plan.options);
        backward_shaper[3-i] = make_reshape3d<backend_tag>(plan.out_shape[i], plan.in_shape[i], comm, plan.options);
    }

    fft0 = one_dim_backend<backend_tag>::make(plan.out_shape[0][this_mpi_rank], plan.fft_direction[0]);
    fft1 = one_dim_backend<backend_tag>::make(plan.out_shape[1][this_mpi_rank], plan.fft_direction[1]);
    fft2 = one_dim_backend<backend_tag>::make(plan.out_shape[2][this_mpi_rank], plan.fft_direction[2]);
}

template<typename backend_tag>
template<typename scalar_type> // complex to complex case
void fft3d<backend_tag>::standard_transform(std::complex<scalar_type> const input[], std::complex<scalar_type> output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                                            std::array<backend_executor*, 3> const executor,
                                            direction dir, scale scaling) const{
    /*
     * The logic is a bit messy, but the objective is:
     * - call all shaper and executor objects in the correct order
     * - assume that any or all of the shapers can be missing, i.e., null unique_ptr()
     * - do not allocate buffers if not needed
     * - never have more than 2 allocated buffers (input and output)
     */
    auto apply_fft = [&](int i, std::complex<scalar_type> data[])
        ->void{
            add_trace name("fft-1d");
            if (dir == direction::forward){
                executor[i]->forward(data);
            }else{
                executor[i]->backward(data);
            }
        };

    int num_active = count_active(shaper);
    int last = get_last_active(shaper);

    if (last < 1){ // no extra buffer case
        add_trace name("less than 1");
        // move input -> output and apply all ffts
        // use either zeroth shaper or simple copy (or nothing in case of in-place transform)
        if (last == 0){
            shaper[0]->apply(input, output, workspace);
        }else if (input != output){
            data_manipulator<location_tag>::copy_n(input, executor[0]->box_size(), output);
        }
        for(int i=0; i<3; i++)
            apply_fft(i, output);
        return;
    }

    // with only one reshape, the temp buffer would be used only if not doing in-place
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();
    if (num_active == 1){ // one active and not shaper 0
        std::complex<scalar_type> *effective_input = output;
        if (input != output){
            data_manipulator<location_tag>::copy_n(input, executor[0]->box_size(), temp_buffer);
            effective_input = temp_buffer;
        }
        for(int i=0; i<last; i++)
            apply_fft(i, effective_input);
        shaper[last]->apply(effective_input, output, workspace);
        for(int i=last; i<3; i++)
            apply_fft(i, output);
        return;
    }

    // with two or more reshapes, the first reshape must move to the temp_buffer and the last must move to output
    int active_shaper = 0;
    if (shaper[0] or input != output){
        if (shaper[0]){
            shaper[0]->apply(input, temp_buffer, workspace);
        }else{
            add_trace name("copy");
            data_manipulator<location_tag>::copy_n(input, executor[0]->box_size(), temp_buffer);
        }
        active_shaper = 1;
    }else{
        // in place transform and shaper[0] is not active
        while(not shaper[active_shaper]){
            // note, at least one shaper must be active, otherwise last will catch it
            apply_fft(active_shaper++, output);
        }
        shaper[active_shaper]->apply(output, temp_buffer, workspace);
        active_shaper += 1;
    }
    apply_fft(active_shaper - 1, temp_buffer); // one reshape was applied above

    for(int i=active_shaper; i<last; i++){
        if (shaper[i])
            shaper[i]->apply(temp_buffer, temp_buffer, workspace);
        apply_fft(i, temp_buffer);
    }
    shaper[last]->apply(temp_buffer, output, workspace);

    for(int i=last; i<3; i++)
        apply_fft(i, output);


    if (scaling != scale::none){
        add_trace name("scale");
        data_manipulator<location_tag>::scale(
            (dir == direction::forward) ? size_outbox() : size_inbox(),
            output, get_scale_factor(scaling));
    }
}
template<typename backend_tag>
template<typename scalar_type> // real to complex case
void fft3d<backend_tag>::standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                                            std::array<typename one_dim_backend<backend_tag>::type*, 3> const executor,
                                            direction, scale scaling) const{
    /*
     * Follows logic similar to the complex-to-complex case but the first shaper and executor will be applied to real data.
     * This is the real-to-complex variant which is possible only for a forward transform,
     * thus the direction parameter is ignored.
     */
    int last = get_last_active(shaper);

    scalar_type* reshaped_input = reinterpret_cast<scalar_type*>(workspace);
    scalar_type const *effective_input = input; // either input or the result of reshape operation 0
    if (shaper[0]){
        shaper[0]->apply(input, reshaped_input, reinterpret_cast<scalar_type*>(workspace + get_max_size(executor)));
        effective_input = reshaped_input;
    }

    if (last < 1){ // no reshapes after 0
        add_trace name("fft-1d x3");
        executor[0]->forward(effective_input, output);
        executor[1]->forward(output);
        executor[2]->forward(output);
        return;
    }

    // if there is messier combination of transforms, then we need internal buffers
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();
    { add_trace name("fft-1d");
    executor[0]->forward(effective_input, temp_buffer);
    }

    for(int i=1; i<last; i++){
        if (shaper[i]){
            shaper[i]->apply(temp_buffer, temp_buffer, workspace);
        }
        add_trace name("fft-1d");
        executor[i]->forward(temp_buffer);
    }
    shaper[last]->apply(temp_buffer, output, workspace);

    for(int i=last; i<3; i++){
        add_trace name("fft-1d");
        executor[i]->forward(output);
    }

    if (scaling != scale::none){
        add_trace name("scale");
        data_manipulator<location_tag>::scale(size_outbox(), output, get_scale_factor(scaling));
    }
}
template<typename backend_tag>
template<typename scalar_type> // complex to real case
void fft3d<backend_tag>::standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                                            std::array<backend_executor*, 3> const executor, direction, scale scaling) const{
    /*
     * Follows logic similar to the complex-to-complex case but the last shaper and executor will be applied to real data.
     * This is the complex-to-real variant which is possible only for a backward transform,
     * thus the direction parameter is ignored.
     */
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();

    if (shaper[0]){
        shaper[0]->apply(input, temp_buffer, workspace);
    }else{
        data_manipulator<location_tag>::copy_n(input, executor[0]->box_size(), temp_buffer);
    }

    for(int i=0; i<2; i++){ // apply the two complex-to-complex ffts
        { add_trace name("fft-1d x3");
        executor[i]->backward(temp_buffer);
        }
        if (shaper[i+1])
            shaper[i+1]->apply(temp_buffer, temp_buffer, workspace);
    }

    // the result of the first two ffts and three reshapes is stored in temp_buffer
    // executor 2 must apply complex to real backward transform
    if (shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        scalar_type* real_buffer = reinterpret_cast<scalar_type*>(temp_buffer);
        { add_trace name("fft-1d");
        executor[2]->backward(temp_buffer, real_buffer);
        }
        shaper[3]->apply(real_buffer, output, reinterpret_cast<scalar_type*>(workspace));
    }else{
        add_trace name("fft-1d");
        executor[2]->backward(temp_buffer, output);
    }

    if (scaling != scale::none){
        add_trace name("scale");
        data_manipulator<location_tag>::scale(size_inbox(), output, get_scale_factor(scaling));
    }
}

#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_fft3d(backend::fftw);
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_fft3d(backend::mkl);
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_fft3d(backend::cufft);
#endif

}
