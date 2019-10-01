/** @class */
/*
    -- HEFFTE (version 0.1) --
       Univ. of Tennessee, Knoxville
       @date
*/
// FFT3d class
#ifndef FFT_FFT3D_H
#define FFT_FFT3D_H

#include <mpi.h>
#include "heffte_utils.h"
#include "heffte_common.h"
#include "reshape3d.h"

#ifdef FFT_FFTW
#define FFT_FFTW3
#endif

namespace HEFFTE_NS {

  /*!
   * The class FFT3d is the main function of HEFFTE library, it is in charge of creating the plans for
   * the 1D FFT computations, as well as the plans for data reshaping. It also calls the appropriate routines
   * for execution and transposition. Objects can be created as follows: new FFT3d<T>(MPI_Comm user_comm)
   * @param user_comm  MPI communicator for the P procs which own the data
   */
 //------------------------------------------------------------------------------


 // Traits to lookup CUFFT data type. Default is cufftDoubleComplex
 #if defined(FFT_CUFFT_A) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

   template <typename T>
   struct cufft_traits
   {
       typedef cufftDoubleComplex complex_type;
   };

   // Specialization for float.
   template <>
   struct cufft_traits<float>
   {
     typedef cufftComplex complex_type;
   };

#else
//------------------------------------------------------------------------------
// Traits to lookup FFTW plan type. Default is double.
 template <typename T>
 struct fftw_traits
 {
     typedef fftw_plan plan_type;
     typedef fftw_complex fftw_complex_data;
 };

 // Specialization for float.
 template <>
 struct fftw_traits<float>
 {
     typedef fftwf_plan plan_type;
     typedef fftwf_complex fftw_complex_data;
 };
 //------------------------------------------------------------------------------
#endif

template <class U>
class FFT3d {
 public:
  MPI_Comm world;
  int scaled, reshapeonly;
  int permute, collective, exchange, packflag, memoryflag;
  int collective_bp, collective_pp;
  int64_t memusage;                 // memory usage in bytes

  enum heffte_memory_type_t mem_type; // memory_type allocations

  int npfast1, npfast2, npfast3;      // size of pencil decomp in fast dim
  int npmid1, npmid2, npmid3;         // ditto for mid dim
  int npslow1, npslow2, npslow3;      // ditto for slow dim
  int npbrick1, npbrick2, npbrick3;   // size of brick decomp in 3 dims

  int ntrial;                            // # of tuning trial runs
  int npertrial;                         // # of FFTs per trial
  int cbest,ebest,pbest;                 // fastest setting for coll,exch,pack
  double computeTime=0.0;
  int cflags[10],eflags[10],pflags[10];  // same 3 settings for each trial
  double besttime;                       // fastest single 3d FFT time
  double setuptime;                      // setup() time after tuning
  double tfft[10];                       // single 3d FFT time for each trial
  double t1d[10];                        // 1d FFT time for each trial
  double treshape[10];                     // total reshape time for each trial
  double treshape1[10],treshape2[10],
    treshape3[10],treshape4[10];             // per-reshape time for each trial

  const char *fft1d;                // name of 1d FFT lib
  const char *precision;            // precision of FFTs, "single" or "double"

  FFT3d(MPI_Comm);
  ~FFT3d();

  template <class T>
  void setup(T* work, int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi, int user_permute, int &user_fftsize, int &user_sendsize, int &user_recvsize);

  template <class T> void setup_memory(T *, T *);
  template <class T> void compute(T *in, T *out, int flag);
  template <class T> void only_1d_ffts(T *, int);
  template <class T> void only_reshapes(T *, T *, int);
  template <class T> void only_one_reshape(T *, T *, int, int);

 private:
  int me,nprocs;
  int setupflag, setup_memory_flag;

  class Memory *memory;
  class Error *error;

  int normnum;                      // # of values to rescale
  U norm;                      // normalization factor for rescaling

  int nprime,nfactor;
  int *primes,*factors;

  int nfast,nmid,nslow;

  int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;
  int out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi;
  int fast_ilo, fast_ihi, fast_jlo, fast_jhi, fast_klo, fast_khi;
  int mid_ilo, mid_ihi, mid_jlo, mid_jhi, mid_klo, mid_khi;
  int slow_ilo, slow_ihi, slow_jlo, slow_jhi, slow_klo, slow_khi;
  int brick_ilo, brick_ihi, brick_jlo, brick_jhi, brick_klo, brick_khi;

  int ipfast1, ipfast2, ipfast3;      // my loc in pencil decomp in fast dim
  int ipmid1, ipmid2, ipmid3;         // ditto for mid dim
  int ipslow1, ipslow2, ipslow3;      // diito for slow dim
  int ipbrick1, ipbrick2, ipbrick3;   // my loc in brick decomp in 3 dims

  int insize, outsize;
  int fastsize, midsize, slowsize, bricksize;
  int fftsize, sendsize, recvsize;

  int inout_layout_same;            // 1 if initial layout = final layout

  // Reshape data structs
  struct Reshape {
    class Reshape3d<U> *reshape3d;
    class Reshape3d<U> *reshape3d_extra;
  };

  Reshape *reshape_prefast, *reshape_fastmid, *reshape_midslow, *reshape_postslow;
  Reshape *reshape_preslow, *reshape_slowmid, *reshape_midfast, *reshape_postfast;
  int reshape_preflag, reshape_postflag;

  U *sendbuf;              // buffer for reshape sends
  U *recvbuf;              // buffer for reshape recvs


  template <class T> void reshape(T *, T *, Reshape *);
  void reshape_forward_create(int &, int &);
  void reshape_inverse_create(int &, int &);
  void deallocate_reshape(Reshape *);
  template <class T> void scale_ffts(T &fft_norm, T *data);


#if defined(FFT_MKL) || defined(FFT_MKL_OMP)
  struct FFT1d {
    int n, length, total;
    DFTI_DESCRIPTOR_HANDLE handle;
  };
#elif defined(FFT_FFTW2)
  struct FFT1d {
    int n, length, total;
    fftw_plan plan_forward;
    fftw_plan plan_backward;
  };
#elif defined(FFT_CUFFT_A) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
  struct FFT1d {
    int n, length, total;
    cufftHandle plan_unique;
  };
#else

    struct FFT1d {
        int n, length, total;
        typename fftw_traits<U>::plan_type plan_forward;
        typename fftw_traits<U>::plan_type plan_backward;
    };

#endif

  // class FFT1d<fftw_plan> *fft_fast,*fft_mid,*fft_slow;
  struct FFT1d *fft_fast, *fft_mid, *fft_slow;

  // private methods
  void deallocate_setup();
  void deallocate_setup_memory();
  int64_t reshape_memory();

  void setup_ffts();
  void perform_ffts(U *, int, FFT1d *);
  void deallocate_ffts();
  int prime_factorable(int);
  void factor(int);
  void procfactors(int, int, int, int &, int &, int &, int &, int &, int &);
  double surfarea(int, int, int, int, int, int);
};

}
#endif
