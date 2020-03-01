/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef FFT_FFT3D_H
#define FFT_FFT3D_H

#include <mpi.h>
#include "heffte_utils.h"
#include "heffte_common.h"
#include "heffte_reshape3d.h"

//#include "heffte_backend_fftw.h"

//#ifdef FFT_FFTW
//#define FFT_FFTW3
//#endif

namespace HEFFTE {

  /*!
   * The class FFT3d is the main function of HEFFTE library, it is in charge of creating the plans for
   * the 1D FFT computations, as well as the plans for data reshaping. It also calls the appropriate routines
   * for execution and transposition. Objects can be created as follows: new FFT3d<T>(MPI_Comm user_comm)
   * @param user_comm  MPI communicator for the P procs which own the data
   */
 //------------------------------------------------------------------------------


 // Traits to lookup CUFFT data type. Default is cufftDoubleComplex
 #if defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

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
  int npslow1_r2c, npslow2_r2c, npslow3_r2c;      // ditto for slow for r2c fft
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

  void setup(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi, int user_permute, int &user_fftsize, int &user_sendsize, int &user_recvsize);

  void setup_r2c(int* N, int* i_lo, int* i_hi, int* o_lo, int* o_hi, int &user_fftsize, int &user_sendsize, int &user_recvsize);


  template <class T> void setup_memory(T *, T *);
  template <class T> void compute(T *in, T *out, int flag);
  template <class T> void compute_r2c(T *in, T *out);
  template <class T> void only_1d_ffts(T *, int);
  template <class T> void only_reshapes(T *, T *, int);
  template <class T> void only_one_reshape(T *, T *, int, int);

  int reshape_preflag, reshape_postflag, reshape_final_grid;

 private:
  int me,nprocs;
  int setupflag, setupflag_r2c, setup_memory_flag;

  class Memory *memory;
  class Error *error;

  int normnum;                      // # of values to rescale
  U norm;                      // normalization factor for rescaling

  std::vector<int> primes, factors;

  int nfast,nmid,nslow;
  int nfast_h; // for R2C transform, ~ half the size of fast direction

  int in_ilo, in_ihi, in_jlo, in_jhi, in_klo, in_khi;
  int out_ilo, out_ihi, out_jlo, out_jhi, out_klo, out_khi;
  int fast_ilo, fast_ihi, fast_jlo, fast_jhi, fast_klo, fast_khi;
  int mid_ilo, mid_ihi, mid_jlo, mid_jhi, mid_klo, mid_khi;
  int slow_ilo, slow_ihi, slow_jlo, slow_jhi, slow_klo, slow_khi;
  int slow_ilo_r2c, slow_ihi_r2c, slow_jlo_r2c, slow_jhi_r2c, slow_klo_r2c, slow_khi_r2c; // needed for r2c fft
  int brick_ilo, brick_ihi, brick_jlo, brick_jhi, brick_klo, brick_khi;

  int ipfast1, ipfast2, ipfast3;      // my loc in pencil decomp in fast dim
  int ipmid1, ipmid2, ipmid3;         // ditto for mid dim
  int ipslow1, ipslow2, ipslow3;      // diito for slow dim
  int ipslow1_r2c, ipslow2_r2c, ipslow3_r2c;      // diito for slow for r2c transform
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

  U *sendbuf;              // buffer for reshape sends
  U *recvbuf;              // buffer for reshape recvs


  template <class T> void reshape(T *, T *, Reshape *);
  void reshape_forward_create(int &, int &);
  void reshape_inverse_create(int &, int &);
  void reshape_r2c_create(int &, int &);


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
  #elif defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)
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

  struct FFT1d *fft_fast, *fft_mid, *fft_slow;

// general methods
  void perform_ffts(U *, int, FFT1d *);
  void perform_ffts_r2c(U *, U *, FFT1d *);

  void deallocate_setup();
  void deallocate_setup_r2c();

  void deallocate_setup_memory();
  int64_t reshape_memory();

  void setup_ffts();
  void setup_ffts_r2c();

  void deallocate_ffts();
  void deallocate_ffts_r2c();
  int prime_factorable(int);
  void factor(int);
  void procfactors(int, int, int, int &, int &, int &, int &, int &, int &);
  double surfarea(int, int, int, int, int, int);
};

}

namespace heffte {

template<typename scalar_type> struct fft_output{ using type = scalar_type; };
template<> struct fft_output<float>{ using type = std::complex<float>; };
template<> struct fft_output<double>{ using type = std::complex<double>; };

//! \brief Work in progress, nothing set in stone.
template<typename backend_tag>
class fft3d{
public:
    using backend_executor = typename one_dim_backend<backend_tag>::type;
    template<typename T> using buffer_container = typename backend::buffer_traits<backend_tag>::template container<T>;

    fft3d(box3d const cinbox, box3d const coutbox, MPI_Comm const);
    fft3d(box3d const box, MPI_Comm const comm) : fft3d(box, box, comm){}

    int size_inbox() const{ return inbox.count(); }
    int size_outbox() const{ return outbox.count(); }

    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[]) const{
        static_assert((std::is_same<input_type, float>::value and is_ccomplex<output_type>::value)
                   or (std::is_same<input_type, double>::value and is_zcomplex<output_type>::value)
                   or (is_ccomplex<input_type>::value and is_ccomplex<output_type>::value)
                   or (is_zcomplex<input_type>::value and is_zcomplex<output_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), forward_shaper, {fft0.get(), fft1.get(), fft2.get()}, direction::forward);
    }

    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[]) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value)
                   or (is_ccomplex<output_type>::value and is_ccomplex<input_type>::value)
                   or (is_zcomplex<output_type>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), backward_shaper, {fft2.get(), fft1.get(), fft0.get()}, direction::backward);
    }

    template<typename input_type>
    std::vector<typename fft_output<input_type>::type> forward(std::vector<input_type> const &input){
        static_assert(std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::cpu>::value,
                      "The std::vector variants of fotward() and backward() fft can be called only with a cpu backend.");
        if (input.size() < size_inbox())
            throw std::invalid_argument("The input vector is smaller than size_inbox(), i.e., not enough entries provided to fill the inbox.");
        std::vector<typename fft_output<input_type>::type> output(size_outbox());
        forward(input.data(), output.data());
        return output;
    }

    template<typename scalar_type>
    std::vector<scalar_type> backward(std::vector<scalar_type> const &input){
        static_assert(std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::cpu>::value,
                      "The std::vector variants of fotward() and backward() fft can be called only with a cpu backend.");
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        if (input.size() < size_outbox())
            throw std::invalid_argument("The input vector is smaller than size_outbox(), i.e., not enough entries provided to fill the outbox.");
        std::vector<scalar_type> result(size_inbox());
        backward(input.data(), result.data());
        return result;
    }

    template<typename scalar_type>
    std::vector<typename define_standard_type<scalar_type>::type::value_type> backward_real(std::vector<scalar_type> const &input){
        static_assert(std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::cpu>::value,
                      "The std::vector variants of fotward() and backward() fft can be called only with a cpu backend.");
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        std::vector<typename define_standard_type<scalar_type>::type::value_type> result(size_inbox());
        backward(input.data(), result.data());
        return result;
    }

private:
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], std::complex<scalar_type> output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir) const; // complex to complex
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir) const; // real to complex
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir) const; // complex to real

    template<typename scalar_type>
    void reshape_stage(std::unique_ptr<reshape3d_base> const &reshape, scalar_type *&src, scalar_type *&dest) const{
        if (not reshape) return;
        reshape->apply(src, dest);
        std::swap(src, dest);
    }

    box3d inbox, outbox;
    std::array<std::unique_ptr<reshape3d_base>, 4> forward_shaper;
    std::array<std::unique_ptr<reshape3d_base>, 4> backward_shaper;

    std::unique_ptr<backend_executor> fft0, fft1, fft2;
};

}

#endif
