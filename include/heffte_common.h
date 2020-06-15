/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFFTE_COMMON_H
#define HEFFFTE_COMMON_H

#include "heffte_geometry.h"
#include "heffte_trace.h"

namespace heffte {

/*!
 * \ingroup fft3d
 * \addtogroup fft3dbackend Backend common wrappers
 *
 * Sub-module that encompasses all backend wrappers and meta data.
 */

/*!
 * \ingroup fft3dbackend
 * \brief Contains internal type-tags.
 *
 * Empty structs do not generate run-time code,
 * but can be used in type checks and overload resolutions at compile time.
 * Such empty classes are called "type-tags".
 */
namespace tag {

/*!
 * \ingroup fft3dbackend
 * \brief Indicates the use of cpu backend and that all input/output data and arrays will be bound to the cpu.
 *
 * Examples of cpu backends are FFTW and MKL.
 */
struct cpu{};
/*!
 * \ingroup fft3dbackend
 * \brief Indicates the use of gpu backend and that all input/output data and arrays will be bound to the gpu device.
 *
 * Example of gpu backend is cuFFT.
 */
struct gpu{};

}

/*!
 * \ingroup fft3dbackend
 * \brief Contains methods for data manipulation either on the CPU or GPU.
 */
template<typename location_tag> struct data_manipulator{};

/*!
 * \ingroup fft3dbackend
 * \brief Specialization for manipulations on the CPU end.
 */
template<> struct data_manipulator<tag::cpu>{
    /*!
     * \brief Wrapper around std::copy_n().
     */
    template<typename source_type, typename destination_type>
    static void copy_n(source_type const source[], size_t num_entries, destination_type destination[]){
        std::copy_n(source, num_entries, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void scale(int num_entries, scalar_type *data, double scale_factor){
        scalar_type alpha = static_cast<scalar_type>(scale_factor);
        for(int i=0; i<num_entries; i++) data[i] *= alpha;
    }
    /*!
     * \brief Complex by real scaling.
     *
     * Depending on the compiler and type of operation, C++ complex numbers can have bad
     * performance compared to float and double operations.
     * Since the scaling factor is always real, scaling can be performed
     * with real arithmetic which is easier to vectorize.
     */
    template<typename precision_type>
    static void scale(int num_entries, std::complex<precision_type> *data, double scale_factor){
        scale<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

/*!
 * \ingroup fft3dbackend
 * \brief Contains type tags and templates metadata for the various backends.
 */
namespace backend {

    /*!
     * \ingroup fft3dbackend
     * \brief Allows to define whether a specific backend interface has been enabled.
     *
     * Defaults to std::false_type, but specializations for each enabled backend
     * will overwrite this to the std::true_type, i.e., define const static bool value
     * which is set to true.
     */
    template<typename tag>
    struct is_enabled : std::false_type{};


    /*!
     * \ingroup fft3dbackend
     * \brief Defines the container for the temporary buffers.
     *
     * Specialization for each backend will define whether the raw-arrays are associated
     * with the CPU or GPU devices and the type of the container that will hold temporary
     * buffers.
     */
    template<typename backend_tag>
    struct buffer_traits{
        //! \brief Tags the raw-array location tag::cpu or tag::gpu, used by the packers.
        using location = tag::cpu;
        //! \brief Defines the container template to use for the temporary buffers in heffte::fft3d.
        template<typename T> using container = std::vector<T>;
    };

    /*!
     * \ingroup fft3dbackend
     * \brief Returns the human readable name of the backend.
     */
    template<typename backend_tag>
    inline std::string name(){ return "unknown"; }
}

/*!
 * \ingroup fft3dmisc
 * \brief Indicates the direction of the FFT (internal use only).
 */
enum class direction {
    //! \brief Forward DFT transform.
    forward,
    //! \brief Inverse DFT transform.
    backward
};

/*!
 * \ingroup fft3dbackend
 * \brief Indicates the structure that will be used by the fft backend.
 */
template<typename> struct one_dim_backend{};

/*!
 * \ingroup fft3dbackend
 * \brief Defines a set of default plan options for a given backend.
 */
template<typename> struct default_plan_options{};

}

#endif   //  #ifndef HEFFTE_COMMON_H
