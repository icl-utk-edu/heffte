/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef FFT_FFT3D_H
#define FFT_FFT3D_H

#include "heffte_reshape3d.h"


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

/*!
 * \brief Defines the relationship between pairs of input-output types in the FFT algorithms.
 *
 * The main class and specializations define a member type that defines the output complex number
 * (with the appropriate precision) for the given input template parameter.
 * This struct handles the complex to complex transforms.
 *
 * \tparam scalar_type defines the input to a discrete Fourier transform algorithm
 */
template<typename scalar_type> struct fft_output{
    //! \brief The output for a complex type is the same type.
    using type = scalar_type;
};
/*!
 * \brief Specialization mapping float to std::complex<float>.
 */
template<> struct fft_output<float>{
    //! \brief The output for a float data is std::complex<float>
    using type = std::complex<float>;
};
/*!
 * \brief Specialization mapping double to std::complex<double>.
 */
template<> struct fft_output<double>{
    //! \brief The output for a double data is std::complex<double>
    using type = std::complex<double>;
};

/*!
 * \brief Indicates the scaling factor to apply on the result of an FFT operation.
 *
 * See the description of heffte::fft3d for details.
 */
enum class scale{
    //! \brief No scale, leave the result unperturbed similar to the FFTW API.
    none,
    //! \brief Apply the full scale of 1 over the number of elements in the world box.
    full,
    //! \brief Symmetric scaling, apply the square-root of the full scaling.
    symmetric
};

/*!
 * \brief Defines the plan for a 3-dimensional discrete Fourier transform performed on a MPI distributed data.
 *
 * \par Overview
 * HeFFTe provides the frontend MPI communication algorithms that sync data movement across the MPI ranks,
 * but relies on a backend implementation of FFT algorithms in one dimension.
 * Multiple backends are supported (currently only the fftw3 library), an available backend has to be
 * specified via a template tag.
 * Forward and backward (inverse) transforms can be performed with different precision using the same
 * heffte::fft3d object so long as the input and output use the same distributed geometry.
 *
 * \par Boxes and Data Distribution
 * HeFFTe assumes that the input and output data is organized in three dimensional boxes,
 * each MPI rank containing one input and one output box (currently those should not be empty).
 * Each box is defined by three low and three high indexes, the indexes contained within a box
 * range from the low to the high inclusively, i.e., the box heffte::box3d({0, 0, 0}, {0, 0, 2})
 * contains three indexes (0, 0, 0), (0, 0, 1) and (0, 0, 2).
 * The following conventions are observed:
 * - global indexing starts at 0
 * - the boxes do not overlap (input can overlap with output, but the individual in/out boxed do not)
 * - input and output boxes may be the same but do not have to overlap
 * - no assumption is being made regarding the organization of ranks and boxes
 *
 * \anchor HeffteFFT3DCompatibleTypes
 * \par Real and Complex Transforms
 * HeFFTe supports forward discrete Fourier transforms that take real or complex entries into complex output.
 * The backward (inverse) transform takes complex data entries back to real or complex data.
 * The precision must always match, e.g., float to std::complex<float>, double and std::complex<double>.
 * <table>
 * <tr><td> Forward transform input </td><td> Forward transform output </td><td/>
 *     <td> Backward transform input </td><td> Backward transform output </td>
 * </tr>
 * <tr><td> float </td><td> std::complex<float> </td><td/>
 *     <td> std::complex<float> </td><td> float </td></tr>
 * <tr><td> double </td><td> std::complex<double> </td><td/>
 *     <td> std::complex<double> </td><td> double </td></tr>
 * <tr><td> std::complex<float> </td><td> std::complex<float> </td><td/>
 *     <td> std::complex<float> </td><td> std::complex<float> </td></tr>
 * <tr><td> std::complex<double> </td><td> std::complex<double> </td><td/>
 *     <td> std::complex<double> </td><td> std::complex<double> </td></tr>
 * </table>
 *
 * \par Complex Numbers
 * By default, HeFFTe works with the C++ native std::complex types,
 * those are supported on both the CPU and GPU devices.
 * However, many libraries provide their own complex types definitions and even though those
 * are usually ABI compatible with the C++ standard types, the compiler treats those as distinct entities.
 * Thus, HeFFTe recognizes the types defined by the backend libraries and additional types can be accepted
 * with a specialization of heffte::is_ccomplex and heffte::is_zcomplex.
 * <table>
 * <tr><td> Backend </tr><td> Type </td><td> C++ Equivalent </td></tr>
 * <tr><td rowspan=2> FFTW3 </td><td> fftwf_complex </td><td> std::complex<float> </td></tr>
 * <tr>                          <td> fftw_complex </td><td> std::complex<double> </td></tr>
 * </table>
 *
 * \par Scaling
 * Applying a forward and inverse DFT operations will leave the result as the original data multiplied
 * by the total number of entries in the world box. Thus, the forward and backward operations are not
 * truly inverses, unless the correct scaling is applied. By default, HeFFTe does not apply scaling,
 * but the methods accept an optional parameter with three different options, see also heffte::scale.
 * <table>
 * <tr><td> Forward </tr><td> Backward-inverse </td></tr>
 * <tr><td> forward(a, b, scaling::none) </tr><td> forward(a, b, scaling::full) </td></tr>
 * <tr><td> forward(a, b, scaling::symmetric) </tr><td> forward(a, b, scaling::symmetric) </td></tr>
 * <tr><td> forward(a, b, scaling::full) </tr><td> forward(a, b, scaling::none) </td></tr>
 * </table>
 */
template<typename backend_tag>
class fft3d{
public:
    //! \brief Alias to the wrapper class for the one dimensional backend library.
    using backend_executor = typename one_dim_backend<backend_tag>::type;
    /*!
     * \brief Alias to the container template associated with the backend.
     *
     * Following C++ RAII style of resource management, HeFFTe uses containers to manage
     * the temporary buffers used during transformation and communication.
     * The CPU backends use std::vector while the GPU backends use (heffte::cuda::vector ... to be implemented).
     */
    template<typename T> using buffer_container = typename backend::buffer_traits<backend_tag>::template container<T>;

    /*!
     * \brief Constructor creating a plan for FFT transform across the given communicator and using the box geometry.
     *
     * \param cinbox is the box for the non-transformed data, i.e., the input for the forward() transform and the output of the backward() transform.
     * \param coutbox is the box for the transformed data, i.e., the output for the forward() transform and the input of the backward() transform.
     * \param comm is the MPI communicator with all ranks that will participate in the FFT.
     */
    fft3d(box3d const cinbox, box3d const coutbox, MPI_Comm const);

    //! \brief Returns the size of the inbox defined in the constructor.
    int size_inbox() const{ return inbox.count(); }
    //! \brief Returns the size of the outbox defined in the constructor.
    int size_outbox() const{ return outbox.count(); }

    /*!
     * \brief Performs a forward Fourier transform using two arrays.
     *
     * \tparam input_type is a type compatible with the input of a forward FFT.
     * \tparam output_type is a type compatible with the output of a forward FFT.
     *
     * The \b input_type and \b output_type must be compatible, see
     * \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is an array of size at least size_inbox() holding the input data corresponding
     *          to the inbox
     * \param output is an array of size at least size_outbox() and will be overwritten with
     *          the result from the transform corresponding to the outbox
     *
     * Note that in the complex-to-complex case, the two arrays can be the same, in which case
     *  the size must be at least std::max(size_inbox(), size_outbox()).
     *  Whether the same or different, padded entities of the arrays will not be accessed.
     */
    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<input_type, float>::value and is_ccomplex<output_type>::value)
                   or (std::is_same<input_type, double>::value and is_zcomplex<output_type>::value)
                   or (is_ccomplex<input_type>::value and is_ccomplex<output_type>::value)
                   or (is_zcomplex<input_type>::value and is_zcomplex<output_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), forward_shaper, {fft0.get(), fft1.get(), fft2.get()}, direction::forward, scaling);
    }

    /*!
     * \brief Vector variant of forward() using input and output std::vector containers.
     *
     * Requires a CPU backend, e.g., backend::fftw, and returns the vectors using only C++ standard types.
     *
     * \tparam input_type is a type compatible with the input of a backward FFT,
     *          see \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is a std::vector with size at least size_inbox() corresponding to the input of forward().
     *
     * \returns std::vector with entries corresponding to the output type and with size equal to size_outbox()
     *          corresponding to the output of forward().
     *
     * \throws std::invalid_argument is the size of the \b input is less than size_inbox().
     *
     * This method allow for a more C++-like calls of the form:
     * \code
     *  std::vector<double> x = ....;
     *  ...
     *  heffte::fft3d fft(inbox, outbox, comm);
     *  auto y = fft.forward(x); // y will be std::vector<std::complex<double>>
     * \endcode
     */
    template<typename input_type>
    buffer_container<typename fft_output<input_type>::type> forward(buffer_container<input_type> const &input, scale scaling = scale::none){
        if (input.size() < size_inbox())
            throw std::invalid_argument("The input vector is smaller than size_inbox(), i.e., not enough entries provided to fill the inbox.");
        buffer_container<typename fft_output<input_type>::type> output(size_outbox());
        forward(input.data(), output.data(), scaling);
        return output;
    }

    /*!
     * \brief Performs a backward Fourier transform using two arrays.
     *
     * \tparam input_type is a type compatible with the input of a backward FFT.
     * \tparam output_type is a type compatible with the output of a backward FFT.
     *
     * The \b input_type and \b output_type must be compatible, see
     * \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is an array of size at least size_outbox() holding the input data corresponding
     *          to the outbox
     * \param output is an array of size at least size_inbox() and will be overwritten with
     *          the result from the transform corresponding to the inbox
     *
     * Note that in the complex-to-complex case, the two arrays can be the same, in which case
     *  the size must be at least std::max(size_inbox(), size_outbox()).
     *  Whether the same or different, padded entities of the arrays will not be accessed.
     */
    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value)
                   or (is_ccomplex<output_type>::value and is_ccomplex<input_type>::value)
                   or (is_zcomplex<output_type>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), backward_shaper, {fft2.get(), fft1.get(), fft0.get()}, direction::backward, scaling);
    }

    /*!
     * \brief Perform complex-to-complex backward FFT using vector API.
     */
    template<typename scalar_type>
    buffer_container<scalar_type> backward(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        if (input.size() < size_outbox())
            throw std::invalid_argument("The input vector is smaller than size_outbox(), i.e., not enough entries provided to fill the outbox.");
        buffer_container<scalar_type> result(size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
    }

    /*!
     * \brief Perform complex-to-real backward FFT using vector API.
     */
    template<typename scalar_type>
    buffer_container<typename define_standard_type<scalar_type>::type::value_type> backward_real(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        buffer_container<typename define_standard_type<scalar_type>::type::value_type> result(size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
    }

private:
    /*!
     * \brief Performs the FFT assuming the input types match the C++ standard.
     *
     * The generic template API will convert the various input types into C++ standards
     * using heffte::convert_to_standart() and will call one of these overloads.
     * The three overloads of standard_transform() perform the operations planned by the constructor.
     *
     * \tparam scalar_type is either float or double, indicating the working precision
     *
     * \param input is the input for the forward or backward transform
     * \param shaper are the four stages of the reshape operations
     * \param executor holds the three stages of the one dimensional FFT algorithm
     * \param dir indicates whether to use the forward or backward method of the executor
     */
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], std::complex<scalar_type> output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale) const; // complex to complex
    /*!
     * \brief Overload to handle the real-to-complex case.
     *
     * The inputs are identical to the complex-to-complex case, except the direction parameter which is ignores
     * since the real-to-complex transform is always forward.
     */
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction, scale) const; // real to complex
    /*!
     * \brief Overload to handle the complex-to-real case.
     *
     * The inputs are identical to the complex-to-complex case, except the direction parameter which is ignores
     * since the complex-to-real transform is always backward.
     */
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                            std::array<std::unique_ptr<reshape3d_base>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale) const; // complex to real

    /*!
     * \brief Performs the reshape operation on the data in base using the helper buffer
     *
     * If the reshape operation is not active, this method does nothing.
     * Otherwise, the reshape is applied from the base to the helper and the two pointers are swapped.
     *
     * \tparam scalar_type is the type of the entries of the two arrays
     *
     * \param reshape is either active or null reshape operation
     * \param base is the source of the reshape on entry, and on exit will hold the reshaped data
     * \param helper is a temporary pointer that will be used for the reshape
     */
    template<typename scalar_type>
    void reshape_stage(std::unique_ptr<reshape3d_base> const &reshape, scalar_type *&base, scalar_type *&helper) const{
        if (not reshape) return;
        reshape->apply(base, helper);
        std::swap(base, helper);
    }

    box3d inbox, outbox;
    double scale_factor;
    std::array<std::unique_ptr<reshape3d_base>, 4> forward_shaper;
    std::array<std::unique_ptr<reshape3d_base>, 4> backward_shaper;

    std::unique_ptr<backend_executor> fft0, fft1, fft2;
};

}

#endif
