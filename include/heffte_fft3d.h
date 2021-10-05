/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_FFT3D_H
#define HEFFTE_FFT3D_H

#include "heffte_reshape3d.h"

/*!
 * \defgroup fft3d Fast Fourier Transform
 *
 * \par HeFFTe C++11 API
 * Encapsulates all classes and method for the C++11 API, most notably:
 * - namespace \ref heffte
 * - class heffte::fft3d
 * - class heffte::fft3d_r2c
 * - class heffte::box3d
 * - enum heffte::scale
 */

/*!
 * \ingroup fft3d
 * \addtogroup fft3dcomplex Complex types
 *
 * By default, HeFFTe works with the C++ native std::complex types,
 * by many backends and client codes favor their own complex types.
 * While the types are binary compatible, i.e., arrays of one type can be
 * safely converted with reinterpret_cast, having to manually make those
 * conversions is far from user-friendly.
 * Thus, HeFFTe also accepts the complex types defined by the enabled backend libraries,
 * and the user can indicate their own custom complex types that are
 * binary compatible with std::complex of single or double precision.
 *
 * In addition, HeFFTe provides definition of the correct input and
 * output types for different transforms, see also \ref HeffteFFT3DCompatibleTypes "compatible types".
 */

/*!
 * \ingroup fft3d
 * \brief Namespace containing all HeFFTe methods and classes.
 */
namespace heffte {

/*!
 * \ingroup fft3dcomplex
 * \brief Defines the relationship between pairs of input-output types in the FFT algorithms.
 *
 * The main class and specializations define a member type that defines the output complex number
 * (with the appropriate precision) for the given input template parameter.
 * This struct handles the complex to complex transforms.
 *
 * \tparam scalar_type defines the input to a discrete Fourier transform algorithm
 */
template<typename scalar_type> struct fft_output{
    //! \brief The output type corresponding to the scalar_type.
    using type = scalar_type;
};
/*!
 * \ingroup fft3dcomplex
 * \brief Specialization mapping float to std::complex<float>.
 */
template<> struct fft_output<float>{
    //! \brief The output for a float data is std::complex<float>
    using type = std::complex<float>;
};
/*!
 * \ingroup fft3dcomplex
 * \brief Specialization mapping double to std::complex<double>.
 */
template<> struct fft_output<double>{
    //! \brief The output for a double data is std::complex<double>
    using type = std::complex<double>;
};

/*!
 * \ingroup fft3dcomplex
 * \brief Defines the relationship between pairs of input-output types in a general transform algorithms.
 *
 * Handles the case where we differentiate between the standard FFT transform and the Cosine Transform.
 */
template<typename scalar_type, typename backend_tag, typename = void>
struct transform_output{
    //! \brief The output type corresponding to the scalar_type and backend_tag.
    //using type = scalar_type;
};
/*!
 * \ingroup fft3dcomplex
 * \brief Specialization for standard FFT.
 */
template<typename scalar_type, typename backend_tag>
struct transform_output<scalar_type, backend_tag, typename std::enable_if<backend::uses_fft_types<backend_tag>::value>::type>{
    //! \brief The output type corresponding to the scalar_type and backend_tag (FFT case).
    using type = typename fft_output<scalar_type>::type;
};
/*!
 * \ingroup fft3dcomplex
 * \brief Specialization for Cosine Transform.
 */
template<typename scalar_type, typename backend_tag>
struct transform_output<scalar_type, backend_tag, typename std::enable_if<not backend::uses_fft_types<backend_tag>::value>::type>{
    //! \brief The output type corresponding to the scalar_type and backend_tag (Cosine Transform case).
    using type = scalar_type;
};

/*!
 * \ingroup fft3d
 * \brief Indicates the scaling factor to apply on the result of an FFT operation.
 *
 * See the description of heffte::fft3d for details.
 */
enum class scale{
    //! \brief No scale, leave the result unperturbed similar to the FFTW API.
    none,
    //! \brief Apply the full scale, divide by the number of elements in the world box.
    full,
    //! \brief Symmetric scaling, apply the square-root of the full scaling.
    symmetric
};

/*!
 * \ingroup fft3d
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
 * <tr><td> Forward transform input </td><td> Forward transform output </td><td></td>
 *     <td> Backward transform input </td><td> Backward transform output </td>
 * </tr>
 * <tr><td> float </td><td> std::complex<float> </td><td></td>
 *     <td> std::complex<float> </td><td> float </td></tr>
 * <tr><td> double </td><td> std::complex<double> </td><td></td>
 *     <td> std::complex<double> </td><td> double </td></tr>
 * <tr><td> std::complex<float> </td><td> std::complex<float> </td><td></td>
 *     <td> std::complex<float> </td><td> std::complex<float> </td></tr>
 * <tr><td> std::complex<double> </td><td> std::complex<double> </td><td></td>
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
 * <tr><td> Backend </td><td> Type </td><td> C++ Equivalent </td></tr>
 * <tr><td rowspan=2> FFTW3 </td><td> fftwf_complex </td><td> std::complex<float> </td></tr>
 * <tr>                          <td> fftw_complex </td><td> std::complex<double> </td></tr>
 * <tr><td rowspan=2> MKL   </td><td> float _Complex </td><td> std::complex<float> </td></tr>
 * <tr>                          <td> double _Complex </td><td> std::complex<double> </td></tr>
 * <tr><td rowspan=2> cuFFT </td><td> cufftComplex </td><td> std::complex<float> </td></tr>
 * <tr>                          <td> cufftDoubleComplex </td><td> std::complex<double> </td></tr>
 * </table>
 *
 * \par Scaling
 * Applying a forward and inverse DFT operations will leave the result as the original data multiplied
 * by the total number of entries in the world box. Thus, the forward and backward operations are not
 * truly inversing, unless the correct scaling is applied. By default, HeFFTe does not apply scaling,
 * but the methods accept an optional parameter with three different options, see also heffte::scale.
 * <table>
 * <tr><td> Forward </td><td> Inverse </td></tr>
 * <tr><td> forward(a, b, scaling::none) </tr><td> forward(a, b, scaling::full) </td></tr>
 * <tr><td> forward(a, b, scaling::symmetric) </tr><td> forward(a, b, scaling::symmetric) </td></tr>
 * <tr><td> forward(a, b, scaling::full) </tr><td> forward(a, b, scaling::none) </td></tr>
 * </table>
 */
template<typename backend_tag, typename index = int>
class fft3d : public backend::device_instance<backend_tag>{
public:
    //! \brief Alias to the wrapper class for the one dimensional backend library.
    using backend_executor = typename one_dim_backend<backend_tag>::executor;
    /*!
     * \brief Alias to the container template associated with the backend.
     *
     * Following C++ RAII style of resource management, HeFFTe uses containers to manage
     * the temporary buffers used during transformation and communication.
     * The CPU backends use std::vector while the GPU backends use heffte::cuda::vector.
     */
    template<typename T> using buffer_container = typename backend::buffer_traits<backend_tag>::template container<T>;
    //! \brief Container of real values corresponding to the complex type T.
    template<typename T> using real_buffer_container = buffer_container<typename define_standard_type<T>::type::value_type>;
    //! \brief Container of the output type corresponding to T, see \ref HeffteFFT3DCompatibleTypes "the table of compatible input and output types".
    template<typename T> using output_buffer_container = buffer_container<typename transform_output<T, backend_tag>::type>;

    /*!
     * \brief Type-tag that is either tag::cpu or tag::gpu to indicate the location of the data.
     */
    using location_tag = typename backend::buffer_traits<backend_tag>::location;

    /*!
     * \brief Constructor creating a plan for FFT transform across the given communicator and using the box geometry.
     *
     * \param inbox is the box for the non-transformed data, i.e., the input for the forward() transform and the output of the backward() transform.
     * \param outbox is the box for the transformed data, i.e., the output for the forward() transform and the input of the backward() transform.
     * \param comm is the MPI communicator with all ranks that will participate in the FFT.
     * \param options is a set of options that define the FFT plan, see heffte::plan_options for details.
     */
    fft3d(box3d<index> const inbox, box3d<index> const outbox, MPI_Comm const comm,
          plan_options const options = default_options<backend_tag>()) :
        fft3d(plan_operations(mpi::gather_boxes(inbox, outbox, comm), -1, set_options<backend_tag>(options)), mpi::comm_rank(comm), comm){
        static_assert(backend::is_enabled<backend_tag>::value, "The requested backend is invalid or has not been enabled.");
    }
    /*!
     * \brief Identical to the other constructor but accepts a GPU stream or queue.
     *
     * \param stream is an initialized GPU stream or queue, the actual type depends on the backend as follows:
     *
     * <table>
     * <tr><td> CPU backend </td><td> void*, the stream is never referenced </td></tr>
     * <tr><td> CUDA backend </td><td> cudaStream_t </td></tr>
     * <tr><td> ROCm backend </td><td> hipStream_t </td></tr>
     * <tr><td> oneAPI backend </td><td> sycl::queue const& </td></tr>
     * </table>
     *
     * In all cases, heFFTe takes a non-owning reference or alias to the stream;
     * deleting the stream is a responsibility of the user but should not be done before the heFFTe object is destroyed.
     * If no stream is provided, heFFTe will use the default CUDA or HIP stream or a default internal SYCL queue;
     * note in the SYCL case the internal queue will create a new SYCL context which is probably not optimal.
     */
    fft3d(typename backend::device_instance<backend_tag>::stream_type gpu_stream,
          box3d<index> const inbox, box3d<index> const outbox, MPI_Comm const comm,
          plan_options const options = default_options<backend_tag>()) :
        fft3d(gpu_stream, plan_operations(mpi::gather_boxes(inbox, outbox, comm), -1, set_options<backend_tag>(options)), mpi::comm_rank(comm), comm){
        static_assert(backend::is_enabled<backend_tag>::value, "The requested backend is invalid or has not been enabled.");
    }


    //! \brief Internal use only, used by the Fortran interface
    fft3d(int il0, int il1, int il2, int ih0, int ih1, int ih2, int io0, int io1, int io2,
          int ol0, int ol1, int ol2, int oh0, int oh1, int oh2, int oo0, int oo1, int oo2,
          MPI_Comm const comm,
          bool use_reorder, int algorithm, bool use_pencils)
        : fft3d(box3d<index>({il0, il1, il2}, {ih0, ih1, ih2}, {io0, io1, io2}),
                box3d<index>({ol0, ol1, ol2}, {oh0, oh1, oh2}, {oo0, oo1, oo2}),
                comm,
                plan_options(use_reorder, static_cast<reshape_algorithm>(algorithm), use_pencils))
    {}
    //! \brief Internal use only, used by the Fortran interface
    fft3d(int il0, int il1, int il2, int ih0, int ih1, int ih2, int io0, int io1, int io2,
          int ol0, int ol1, int ol2, int oh0, int oh1, int oh2, int oo0, int oo1, int oo2,
          MPI_Comm const comm)
        : fft3d(box3d<index>({il0, il1, il2}, {ih0, ih1, ih2}, {io0, io1, io2}),
                box3d<index>({ol0, ol1, ol2}, {oh0, oh1, oh2}, {oo0, oo1, oo2}),
                comm)
    {}
    //! \brief Internal use only, used by the Fortran interface
    fft3d(int il0, int il1, int il2, int ih0, int ih1, int ih2,
          int ol0, int ol1, int ol2, int oh0, int oh1, int oh2,
          MPI_Comm const comm)
        : fft3d(box3d<index>({il0, il1, il2}, {ih0, ih1, ih2}), box3d<index>({ol0, ol1, ol2}, {oh0, oh1, oh2}), comm)
    {}

    //! \brief Returns the size of the inbox defined in the constructor.
    long long size_inbox() const{ return pinbox->count(); }
    //! \brief Returns the size of the outbox defined in the constructor.
    long long size_outbox() const{ return poutbox->count(); }
    //! \brief Returns the inbox.
    box3d<index> inbox() const{ return *pinbox; }
    //! \brief Returns the outbox.
    box3d<index> outbox() const{ return *poutbox; }

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
     * \param scaling defines the type of scaling to apply (default no-scaling).
     *
     * Note that in the complex-to-complex case, the two arrays can be the same, in which case
     *  the size must be at least std::max(size_inbox(), size_outbox()).
     *  Whether the same or different, padded entities of the arrays will not be accessed.
     */
    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert(backend::check_types<backend_tag, input_type, output_type>::value,
                      "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), forward_shaper, {fft0.get(), fft1.get(), fft2.get()}, direction::forward, scaling);
    }

    /*!
     * \brief An overload utilizing a user-allocated workspace buffer.
     *
     * HeFFTe requires additional buffers to for various MPI operations, e.g., pack-send-receive-unpack.
     * In the standard overload, the extra memory will be allocated during the call to forward()
     * and released right after.
     * However, allocating and deallocation of large buffers can have a measurable negative effect on performance.
     * Optionally, the use can allocate the workspace buffer externally and pass it into the HeFFTe calls.
     *
     * The workspace buffer must have size equal to size_workspace() and measured in number of complex scalars,
     * e.g., std::complex<float> or std::complex<double> for single and double precision respectively.
     */
    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], output_type workspace[], scale scaling = scale::none) const{
        static_assert(backend::check_types<backend_tag, input_type, output_type>::value,
                      "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), convert_to_standard(workspace),
                           forward_shaper, {fft0.get(), fft1.get(), fft2.get()}, direction::forward, scaling);
    }

    /*!
     * \brief Vector variant of forward() using input and output buffer_container classes.
     *
     * Returns either std::vector or heffte::cuda:vector using only the C++ standard types.
     *
     * \tparam input_type is a type compatible with the input of a backward FFT,
     *          see \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is a std::vector or heffte::cuda::vector with size at least size_inbox() corresponding to the input of forward().
     * \param scaling defines the type of scaling to apply (default no-scaling).
     *
     * \returns std::vector or heffte::cuda::vector with entries corresponding to the output type and with size equal to size_outbox()
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
    output_buffer_container<input_type> forward(buffer_container<input_type> const &input, scale scaling = scale::none){
        if (input.size() < static_cast<size_t>(size_inbox()))
            throw std::invalid_argument("The input vector is smaller than size_inbox(), i.e., not enough entries provided to fill the inbox.");
        auto output = make_buffer_container<typename transform_output<input_type, backend_tag>::type>(this->stream(), size_outbox());
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
     * \param scaling defines the type of scaling to apply (default no-scaling)
     *
     * Note that in the complex-to-complex case, the two arrays can be the same, in which case
     *  the size must be at least std::max(size_inbox(), size_outbox()).
     *  Whether the same or different, padded entities of the arrays will not be accessed.
     */
    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert(backend::check_types<backend_tag, output_type, input_type>::value,
                      "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), backward_shaper, {fft2.get(), fft1.get(), fft0.get()}, direction::backward, scaling);
    }

    /*!
     * \brief Overload with user-provided workspace buffer, see the corresponding overload of forward().
     */
    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], input_type workspace[], scale scaling = scale::none) const{
        static_assert(backend::check_types<backend_tag, output_type, input_type>::value,
                      "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), convert_to_standard(workspace),
                           backward_shaper, {fft2.get(), fft1.get(), fft0.get()}, direction::backward, scaling);
    }

    /*!
     * \brief Perform complex-to-complex backward FFT using vector API.
     */
    template<typename scalar_type>
    buffer_container<scalar_type> backward(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        if (input.size() < static_cast<size_t>(size_outbox()))
            throw std::invalid_argument("The input vector is smaller than size_outbox(), i.e., not enough entries provided to fill the outbox.");
        auto result = make_buffer_container<scalar_type>(this->stream(), size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
    }

    /*!
     * \brief Perform complex-to-real backward FFT using vector API (truncates the complex part).
     */
    template<typename scalar_type>
    real_buffer_container<scalar_type> backward_real(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        auto result = make_buffer_container<typename define_standard_type<scalar_type>::type::value_type>(this->stream(), size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
    }

    /*!
     * \brief Returns the scale factor for the given scaling.
     */
    double get_scale_factor(scale scaling) const{
        if (backend::uses_fft_types<backend_tag>::value){
            return (scaling == scale::symmetric) ? std::sqrt(scale_factor) : scale_factor;
        }else{
            return (scaling == scale::symmetric) ? std::sqrt(scale_factor / 64.0) : scale_factor / 64.0;
        }
    }

    /*!
     * \brief Returns the workspace size that will be used, size is measured in complex numbers.
     */
    size_t size_workspace() const{
        // the last junk of (fft0->box_size() + 1) / 2 is used only when doing complex-to-real backward transform
        // maybe update the API to call for different size buffers for different complex/real types
        return std::max(get_workspace_size(forward_shaper), get_workspace_size(backward_shaper))
               + get_max_size(std::array<backend_executor*, 3>{fft0.get(), fft1.get(), fft2.get()})
               + ((backward_shaper[3]) ? (fft0->box_size() + 1) / 2 : 0);
    }
    //!\brief Returns the size used by the communication workspace buffers (internal use).
    size_t size_comm_buffers() const{ return std::max(get_workspace_size(forward_shaper), get_workspace_size(backward_shaper)); }

private:
    /*!
     * \brief Initialize the class using the provided plan and communicator.
     *
     * This constructor is private to prevent direct call from the user.
     * The user can provide higher-level logic related to the geometry and the plan is created
     * after analysis of the geometry, but the analysis is performed externally to the class.
     * The separation of the constructor and the geometry analysis gives us:
     * - better code organization, e.g., we don't clutter the fft class with messy analysis logic
     *   and we do not have to repeat the logic again for the fft3d_r2c class
     * - simpler interface, e.g., the user provides only high-level geometry data
     *   and we (internally) ensure that the plan is built on the correct communicator
     * - stricter enforcement of const, e.g., we can have const variables that require multiple
     *   analysis steps before they can be initialized.
     *
     * \param plan is the description of the input and output shapes on each stage of the transform
     *             and the direction of the 1-D ffts
     * \param this_mpi_rank is the rank of this mpi process, i.e., mpi::comm_rank(comm)
     * \param comm is the communicator operating on the data
     */
    fft3d(logic_plan3d<index> const &plan, int const this_mpi_rank, MPI_Comm const comm);

    //! \brief Same as the other case but accepts the gpu_stream too.
    fft3d(typename backend::device_instance<backend_tag>::stream_type gpu_stream,
          logic_plan3d<index> const &plan, int const this_mpi_rank, MPI_Comm const comm);

    //! \brief Setup the executors and the reshapes.
    void setup(logic_plan3d<index> const &plan, MPI_Comm const comm);

    /*!
     * \brief Performs the FFT assuming the input types match the C++ standard.
     *
     * The generic template API will convert the various input types into C++ standards
     * using heffte::convert_to_standard() and will call one of these overloads.
     * The three overloads of standard_transform() perform the operations planned by the constructor.
     *
     * \tparam scalar_type is either float or double, indicating the working precision
     *
     * \param input is the input for the forward or backward transform
     * \param output is the output for the forward or backward transform
     * \param workspace is the pre-allocated buffer with size at least size_workspace()
     * \param shaper are the four stages of the reshape operations
     * \param executor holds the three stages of the one dimensional FFT algorithm
     * \param dir indicates whether to use the forward or backward method of the executor
     * \param scaling is the type of scaling to apply to the output data
     */
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], scalar_type output[], scalar_type workspace[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale scaling) const; // complex to complex
    //! \brief Overload that allocates and deallocates the workspace.
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], scalar_type output[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale scaling) const{
        auto workspace = make_buffer_container<typename transform_output<scalar_type, backend_tag>::type>(this->stream(), size_workspace());
        standard_transform(input, output, workspace.data(), shaper, executor, dir, scaling);
    }
    /*!
     * \brief Overload to handle the real-to-complex case.
     *
     * The inputs are identical to the complex-to-complex case, except the direction parameter which is ignores
     * since the real-to-complex transform is always forward.
     */
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                            std::complex<scalar_type> workspace[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction, scale) const; // real to complex
    //! \brief Overload that allocates and deallocates the workspace.
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale scaling) const{
        auto workspace = make_buffer_container<std::complex<scalar_type>>(this->stream(), size_workspace());
        standard_transform(input, output, workspace.data(), shaper, executor, dir, scaling);
    }
    /*!
     * \brief Overload to handle the complex-to-real case.
     *
     * The inputs are identical to the complex-to-complex case, except the direction parameter which is ignores
     * since the complex-to-real transform is always backward.
     */
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                            std::complex<scalar_type> workspace[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale) const; // complex to real
    //! \brief Overload that allocates and deallocates the workspace.
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                            std::array<backend_executor*, 3> const executor, direction dir, scale scaling) const{
        auto workspace = make_buffer_container<std::complex<scalar_type>>(this->stream(), size_workspace());
        standard_transform(input, output, workspace.data(), shaper, executor, dir, scaling);
    }
    //! \brief Applies the scaling factor to the data.
    template<typename scalar_type>
    void apply_scale(direction dir, scale scaling, scalar_type data[]) const{
        if (scaling != scale::none){
            add_trace name("scale");
            #ifdef Heffte_ENABLE_MAGMA
            if (std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::gpu>::value){
                hmagma.scal((dir == direction::forward) ? size_outbox() : size_inbox(), get_scale_factor(scaling), data);
                return;
            }
            #endif
            data_scaling::apply(
                this->stream(),
                (dir == direction::forward) ? size_outbox() : size_inbox(),
                data, get_scale_factor(scaling));
        }
    }

    std::unique_ptr<box3d<index>> pinbox, poutbox; // inbox/output for this process
    double scale_factor;
    std::array<std::unique_ptr<reshape3d_base<index>>, 4> forward_shaper;
    std::array<std::unique_ptr<reshape3d_base<index>>, 4> backward_shaper;

    std::unique_ptr<backend_executor> fft0, fft1, fft2;
    #ifdef Heffte_ENABLE_MAGMA
    gpu::magma_handle<typename backend::buffer_traits<backend_tag>::location> hmagma;
    #endif
};

/*!
 * \ingroup fft3d
 * \brief Alias of heffte::fft3d to be used for a two dimensional problem.
 *
 * The internal logic of heFFTe is capable of recognizing directions with only a single indexes
 * and ignoring redundant communication. Thus, a two dimensional transform
 * is just an alias for the three dimensional one with heffte::box2d as input (which is also an alias).
 */
template<typename backend_tag, typename index = int>
using fft2d = fft3d<backend_tag, index>;

/*!
 * \ingroup fft3d
 * \brief Alias of heffte::fft3d to be more expressive when using Sine and Cosine transforms.
 *
 * The "r" in front of the name relates to the real type of the transform and differentiates the name from std::transform
 * in case the use includes the heffte namespace directly.
 */
template<typename backend_tag, typename index = int>
using rtransform = fft3d<backend_tag, index>;

/*!
 * \ingroup fft3d
 * \brief Factory method that auto-detects the index type based on the box.
 */
template<typename backend_tag, typename index>
fft3d<backend_tag, index> make_fft3d(box3d<index> const inbox, box3d<index> const outbox, MPI_Comm const comm,
                                     plan_options const options = default_options<backend_tag>()){
    static_assert(std::is_same<index, int>::value or std::is_same<index, long long>::value,
                  "heFFTe works with 'int' and 'long long' indexing only");
    static_assert(backend::is_enabled<backend_tag>::value,
                  "the backend_tag is not valid, perhaps it needs to be enabled in the build system");
    return fft3d<backend_tag, index>(inbox, outbox, comm, options);
}

}

#endif
