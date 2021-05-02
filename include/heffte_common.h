/*
    -- heFFTe --
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
 * \brief Common data-transfer and scaling operations.
 */
namespace data_manipulator {
    //! \brief Wrapper around std::copy_n().
    template<typename source_type, typename destination_type>
    void copy_n(void*, source_type const source[], size_t num_entries, destination_type destination[]){
        std::copy_n(source, num_entries, destination);
    }
    //! \brief Wrapper around std::copy_n().
    template<typename source_type, typename destination_type>
    void copy_n(source_type const source[], size_t num_entries, destination_type destination[]){
        std::copy_n(source, num_entries, destination);
    }
}

/*!
 * \ingroup fft3dbackend
 * \brief Contains type tags and templates metadata for the various backends.
 */
namespace backend {

    /*!
     * \ingroup hefftecuda
     * \brief Type-tag for the cuFFT backend
     */
    struct cufft{};

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
    template<typename backend_tag, typename std::enable_if<is_enabled<backend_tag>::value, void*>::type = nullptr>
    struct buffer_traits{
        //! \brief Tags the raw-array location tag::cpu or tag::gpu, used by the packers.
        using location = tag::cpu;
        //! \brief Defines the container template to use for the temporary buffers in heffte::fft3d.
        template<typename T> using container = std::vector<T>;
    };

    /*!
     * \ingroup fft3dbackend
     * \brief Struct that specializes to true type if the location of the backend is on the gpu (false type otherwise).
     */
    template<typename backend_tag, typename = void>
    struct uses_gpu : std::false_type{};

    /*!
     * \ingroup fft3dbackend
     * \brief Specialization for the on-gpu case.
     */
    template<typename backend_tag>
    struct uses_gpu<backend_tag,
                    typename std::enable_if<std::is_same<typename buffer_traits<backend_tag>::location, tag::gpu>::value, void>::type>
    : std::true_type{};

    /*!
     * \ingroup fft3dbackend
     * \brief Returns the human readable name of the backend.
     */
    template<typename backend_tag>
    inline std::string name(){ return "unknown"; }

    /*!
     * \ingroup hefftecuda
     * \brief Returns the human readable name of the cuFFT backend.
     */
    template<> inline std::string name<cufft>(){ return "cufft"; }

    /*!
     * \ingroup fft3dbackend
     * \brief Holds the auxiliary variables needed by each backend.
     *
     * The idea is similar to <a href="https://en.cppreference.com/w/cpp/language/crtp">CRTP</a>
     * heffte::fft3d and heffte::fft3d_r2c inherit from this class and specializations based
     * on the backend-tag can define a different set of internal variables.
     * Specifically, this is used to store the sycl::queue used by the DPC++ backend.
     */
    template<typename backend_tag>
    struct auxiliary_variables{
        //! \brief Empty constructor.
        auxiliary_variables(){}
        //! \brief Empty constructor.
        auxiliary_variables(void*){}
        //! \brief Default destructor.
        virtual ~auxiliary_variables() = default;
        //! \brief Returns the nullptr.
        void* gpu_queue(){ return nullptr; }
        //! \brief Returns the nullptr (const case).
        void* gpu_queue() const{ return nullptr; }
        //! \brief The type for the internal stream, the cpu uses just a void pointer.
        using queue_type = void*;
    };

    /*!
     * \ingroup fft3dbackend
     * \brief Defines inverse mapping from the location tag to a default backend tag.
     *
     * Used in conjunction with backend::auxiliary_variables,
     * by default the auxiliary_variables are not dependent on the tag so the map makes no difference.
     */
    template<typename location_tag> struct from_location{
        //! \brief Defaults to the same label.
        using type = location_tag;
    };
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
