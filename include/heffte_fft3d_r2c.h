/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_FFT3D_R2C_H
#define HEFFTE_FFT3D_R2C_H

#include "heffte_fft3d.h"

namespace heffte {

/*!
 * \ingroup fft3d
 * \brief Similar to heffte::fft3d, but computed fewer redundant coefficients when the input is real.
 *
 * \par Overview
 * Given real input data, there is no unambiguous way to distinguish between the positive and negative
 * direction in the complex plane; therefore, by an argument of symmetry, all complex output must come
 * in conjugate pairs. The heffte::fft3d computes both numbers for each conjugate pair,
 * this class aims at computing fewer redundant coefficients and thus reducing both flops and data movement.
 * This is achieved by selecting one of the three dimensions and the data is shortened in that dimensions
 * to contain only the unique (non-conjugate) coefficients.
 *
 * \par Boxes and Data Distribution
 * Similar to heffte::fft3d the data is organized in boxes using the heffte::box3d structs;
 * however, in the real-to-complex case the global input and output domains do not match.
 * If the original data sits in a box {0, 0, 0}, {x, y, z}, then depending on the dimensions
 * chosen for the shortening, the output data will form the box:
 * \code
 *  {{0, 0, 0}, {x/2 + 1, y,       z}}        // if chosen dimension 0
 *  {{0, 0, 0}, {x,       y/2 + 1, z}}        // if chosen dimension 1
 *  {{0, 0, 0}, {x,       y,       z/2 + 1}}  // if chosen dimension 2
 * // note that x/2 indicates the standard C++ integer division
 * \endcode
 * Thus, the union of the inboxes across all MPI ranks must add up to the global input box,
 * and the union of the outboxes must add up to the shortened global box.
 *
 * \par Compatible Types
 * The real-to-complex variant does not support the cases when the input is complex,
 * the supported types are the ones with real input in
 * \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
 */
template<typename backend_tag>
class fft3d_r2c{
public:
    //! \brief FFT executor for the complex-to-complex dimensions.
    using backend_executor_c2c = typename one_dim_backend<backend_tag>::type;
    //! \brief FFT executor for the real-to-complex dimension.
    using backend_executor_r2c = typename one_dim_backend<backend_tag>::type_r2c;
    /*!
     * \brief Type-tag that is either tag::cpu or tag::gpu to indicate the location of the data.
     */
    using location_tag = typename backend::buffer_traits<backend_tag>::location;

    /*!
     * \brief Alias to the container template associated with the backend (allows for RAII memory management).
     */
    template<typename T> using buffer_container = typename backend::buffer_traits<backend_tag>::template container<T>;
    //! \brief Container of real values corresponding to the complex type T.
    template<typename T> using real_buffer_container = buffer_container<typename define_standard_type<T>::type::value_type>;
    //! \brief Container of the output type corresponding to T, see \ref HeffteFFT3DCompatibleTypes "the table of compatible input and output types".
    template<typename T> using output_buffer_container = buffer_container<typename fft_output<T>::type>;

    /*!
     * \brief Constructor creating a plan for FFT transform across the given communicator and using the box geometry.
     *
     * \param inbox is the box for the non-transformed data, i.e., the input for the forward() transform and the output of the backward() transform.
     * \param outbox is the box for the transformed data, i.e., the output for the forward() transform and the input of the backward() transform.
     * \param r2c_direction indicates the direction where the total set of coefficients will be reduced to hold only the non-conjugate pairs;
     *        selecting a dimension with odd number of indexes will result in (slightly) smaller final data set.
     * \param comm is the MPI communicator with all ranks that will participate in the FFT.
     * \param options is a set of options that define the FFT plan, see heffte::plan_options for details.
     */
    fft3d_r2c(box3d const inbox, box3d const outbox, int r2c_direction, MPI_Comm const comm,
              plan_options const options = default_options<backend_tag>()) :
        fft3d_r2c(plan_operations(mpi::gather_boxes(inbox, outbox, comm), r2c_direction,
                                  #ifdef Heffte_ENABLE_ROCM
                                  (std::is_same<backend_tag, backend::rocfft>::value) ? force_reorder(options) :
                                  #endif
                                  options),
                  mpi::comm_rank(comm), comm){
        assert(r2c_direction == 0 or r2c_direction == 1 or r2c_direction == 2);
        static_assert(backend::is_enabled<backend_tag>::value, "The requested backend is invalid or has not been enabled.");
    }

    //! \brief Internal use only, used by the Fortran interface
    fft3d_r2c(int il0, int il1, int il2, int ih0, int ih1, int ih2, int io0, int io1, int io2,
              int ol0, int ol1, int ol2, int oh0, int oh1, int oh2, int oo0, int oo1, int oo2,
              int r2c_direction, MPI_Comm const comm,
              bool use_reorder, bool use_alltoall, bool use_pencils)
        : fft3d_r2c(box3d({il0, il1, il2}, {ih0, ih1, ih2}, {io0, io1, io2}),
                    box3d({ol0, ol1, ol2}, {oh0, oh1, oh2}, {oo0, oo1, oo2}),
                    r2c_direction, comm,
                plan_options(use_reorder, use_alltoall, use_pencils))
    {}
    //! \brief Internal use only, used by the Fortran interface
    fft3d_r2c(int il0, int il1, int il2, int ih0, int ih1, int ih2, int io0, int io1, int io2,
              int ol0, int ol1, int ol2, int oh0, int oh1, int oh2, int oo0, int oo1, int oo2,
              int r2c_direction, MPI_Comm const comm)
        : fft3d_r2c(box3d({il0, il1, il2}, {ih0, ih1, ih2}, {io0, io1, io2}),
                    box3d({ol0, ol1, ol2}, {oh0, oh1, oh2}, {oo0, oo1, oo2}),
                    r2c_direction, comm)
    {}
    //! \brief Internal use only, used by the Fortran interface
    fft3d_r2c(int il0, int il1, int il2, int ih0, int ih1, int ih2,
              int ol0, int ol1, int ol2, int oh0, int oh1, int oh2,
              int r2c_direction, MPI_Comm const comm)
        : fft3d_r2c(box3d({il0, il1, il2}, {ih0, ih1, ih2}), box3d({ol0, ol1, ol2}, {oh0, oh1, oh2}), r2c_direction, comm)
    {}

    //! \brief Returns the size of the inbox defined in the constructor.
    int size_inbox() const{ return pinbox->count(); }
    //! \brief Returns the size of the outbox defined in the constructor.
    int size_outbox() const{ return poutbox->count(); }
    //! \brief Returns the inbox.
    box3d inbox() const{ return *pinbox; }
    //! \brief Returns the outbox.
    box3d outbox() const{ return *poutbox; }
    //! \brief Returns the workspace size that will be used, size is measured in complex numbers.
    size_t size_workspace() const{
        return std::max(get_workspace_size(forward_shaper), get_workspace_size(backward_shaper))
               + get_max_size(executor_r2c, executor);

    }
    //! \brief Returns the size used by the communication workspace buffers (internal use).
    size_t size_comm_buffers() const{ return std::max(get_workspace_size(forward_shaper), get_workspace_size(backward_shaper)); }

    /*!
     * \brief Performs a forward Fourier transform using two arrays.
     *
     * \tparam input_type is either float or double type.
     * \tparam output_type is a type compatible with the output of a forward FFT,
     *         see \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is an array of size at least size_inbox() holding the input data corresponding
     *          to the inbox
     * \param output is an array of size at least size_outbox() and will be overwritten with
     *          the result from the transform corresponding to the outbox
     * \param scaling defines the type of scaling to apply (default no-scaling).
     */
    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<input_type, float>::value and is_ccomplex<output_type>::value)
                   or (std::is_same<input_type, double>::value and is_zcomplex<output_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), scaling);
    }

    //! \brief Overload utilizing a user provided buffer.
    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], output_type workspace[], scale scaling = scale::none) const{
        static_assert((std::is_same<input_type, float>::value and is_ccomplex<output_type>::value)
                   or (std::is_same<input_type, double>::value and is_zcomplex<output_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), convert_to_standard(workspace), scaling);
    }

    /*!
     * \brief Vector variant of forward() using input and output buffer_container classes.
     *
     * Returns either std::vector or heffte::cuda:vector using only the C++ standard types.
     * Allows for more C++-ish calls and RAII memory management, see the heffte::fft3d equivalent.
     *
     * \tparam input_type is either float or double.
     *
     * \param input is a std::vector or heffte::cuda::vector with size at least size_inbox() corresponding to the input of forward().
     * \param scaling defines the type of scaling to apply (default no-scaling).
     *
     * \returns std::vector or heffte::cuda::vector with entries corresponding to the output type and with size equal to size_outbox()
     *          corresponding to the output of forward().
     *
     * \throws std::invalid_argument is the size of the \b input is less than size_inbox().
     */
    template<typename input_type>
    output_buffer_container<input_type> forward(buffer_container<input_type> const &input, scale scaling = scale::none){
        if (input.size() < static_cast<size_t>(size_inbox()))
            throw std::invalid_argument("The input vector is smaller than size_inbox(), i.e., not enough entries provided to fill the inbox.");
        static_assert(std::is_same<input_type, float>::value or std::is_same<input_type, double>::value,
                      "The input to forward() must be real, i.e., either float or double.");
        buffer_container<typename fft_output<input_type>::type> output(size_outbox());
        forward(input.data(), output.data(), scaling);
        return output;
    }

    /*!
     * \brief Performs a backward Fourier transform using two arrays.
     *
     * \tparam input_type is either float or double.
     * \tparam output_type is a type compatible with the output of a backward FFT,
     *                     see \ref HeffteFFT3DCompatibleTypes "the table of compatible types".
     *
     * \param input is an array of size at least size_outbox() holding the input data corresponding
     *          to the outbox
     * \param output is an array of size at least size_inbox() and will be overwritten with
     *          the result from the transform corresponding to the inbox
     * \param scaling defines the type of scaling to apply (default no-scaling)
     */
    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), scaling);
    }

    //! \brief Overload utilizing a user provided buffer.
    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], input_type workspace[], scale scaling = scale::none) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standard(input), convert_to_standard(output), convert_to_standard(workspace), scaling);
    }

    /*!
     * \brief Variant of backward() that uses buffer_container for RAII style of resource management.
     */
    template<typename scalar_type>
    real_buffer_container<scalar_type> backward(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        buffer_container<typename define_standard_type<scalar_type>::type::value_type> result(size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
    }

    /*!
     * \brief Returns the scale factor for the given scaling.
     */
    double get_scale_factor(scale scaling) const{ return (scaling == scale::symmetric) ? std::sqrt(scale_factor) : scale_factor; }

private:
    //! \brief Same as in the fft3d case.
    fft3d_r2c(logic_plan3d const &plan, int const this_mpi_rank, MPI_Comm const comm);

    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[], scale scaling) const{
        buffer_container<std::complex<scalar_type>> workspace(size_workspace());
        standard_transform(input, output, workspace.data(), scaling);
    }
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[], scale scaling) const{
        buffer_container<std::complex<scalar_type>> workspace(size_workspace());
        standard_transform(input, output, workspace.data(), scaling);
    }

    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[], std::complex<scalar_type> workspace[], scale) const;
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[], std::complex<scalar_type> workspace[], scale) const;

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
            data_manipulator<location_tag>::scale(
                (dir == direction::forward) ? size_outbox() : size_inbox(),
                data, get_scale_factor(scaling));
        }
    }

    std::unique_ptr<box3d> pinbox, poutbox;
    double scale_factor;
    std::array<std::unique_ptr<reshape3d_base>, 4> forward_shaper;
    std::array<std::unique_ptr<reshape3d_base>, 4> backward_shaper;

    std::unique_ptr<backend_executor_r2c> executor_r2c;
    std::array<std::unique_ptr<backend_executor_c2c>, 2> executor;
    #ifdef Heffte_ENABLE_MAGMA
    gpu::magma_handle<typename backend::buffer_traits<backend_tag>::location> hmagma;
    #endif
};

/*!
 * \ingroup fft3d
 * \brief Alias of heffte::fft2d to be used for a two dimensional problem.
 */
template<typename backend_tag>
using fft2d_r2c = fft3d_r2c<backend_tag>;

}

#endif
