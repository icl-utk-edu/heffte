/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_ONEAPI_H
#define HEFFTE_BACKEND_ONEAPI_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_ONEAPI

#include "heffte_backend_vector.h"
#include "heffte_backend_mkl.h"

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>
#include "oneapi/mkl.hpp"

#ifdef Heffte_ENABLE_MAGMA
//#include "heffte_magma_helpers.h"
#endif

/*!
 * \ingroup fft3d
 * \addtogroup heffteoneapi Backend oneAPI
 *
 * Wrappers and template specializations related to the oneMKL backend.
 * Requires CMake option:
 * \code
 *  -D Heffte_ENABLE_ONEAPI=ON
 * \endcode
 *
 * In addition to the oneMKL wrappers, this also includes a series of kernels
 * for packing/unpacking/scaling the data, as well as a simple container
 * that wraps around SYCL arrays for RAII style of resource management.
 */

namespace heffte{

/*!
 * \ingroup heffteoneapi
 * \brief SYCL/DPC++ specific methods, vector-like container, error checking, etc.
 */
namespace oneapi {
    /*!
     * \ingroup heffteoneapi
     * \brief Memory management operation specific to SYCL/DPC++, see gpu::device_vector.
     */
    struct memory_manager{
        //! \brief Allocate memory.
        static void* allocate(size_t num_bytes);
        //! \brief Free memory.
        static void free(void *pntr);
        //! \brief Send data to the device.
        static void host_to_device(void const *source, size_t num_bytes, void *destination);
        //! \brief Copy within the device.
        static void device_to_device(void const *source, size_t num_bytes, void *destination);
        //! \brief Receive from the device.
        static void device_to_host(void const *source, size_t num_bytes, void *destination);
    };
}

namespace gpu {
    /*!
     * \ingroup heffteoneapi
     * \brief Device vector for the oneAPI backends.
     */
    template<typename scalar_type>
    using vector = device_vector<scalar_type, oneapi::memory_manager>;

    /*!
     * \ingroup heffteoneapi
     * \brief Transfer helpers for the oneAPI backends.
     */
    using transfer = device_transfer<oneapi::memory_manager>;

};

/*!
 * \ingroup heffteoneapi
 * \brief Cuda specific methods, vector-like container, error checking, etc.
 */
namespace oneapi {

    /*!
     * \ingroup heffteoneapi
     * \brief Convert real numbers to complex when both are located on the GPU device.
     *
     * Launches a SYCL/DPC++ kernel.
     */
    template<typename precision_type, typename index>
    void convert(index num_entries, precision_type const source[], std::complex<precision_type> destination[]);
    /*!
     * \ingroup heffteoneapi
     * \brief Convert complex numbers to real when both are located on the GPU device.
     *
     * Launches a SYCL/DPC++ kernel.
     */
    template<typename precision_type, typename index>
    void convert(index num_entries, std::complex<precision_type> const source[], precision_type destination[]);

    /*!
     * \ingroup heffteoneapi
     * \brief Scales real data (double or float) by the scaling factor.
     */
    template<typename scalar_type, typename index>
    void scale_data(index num_entries, scalar_type *data, double scale_factor);
}

/*!
 * \ingroup heffteoneapi
 * \brief Data manipulations on the GPU end.
 */
template<> struct data_manipulator<tag::gpu>{
    /*!
     * \brief Equivalent to std::copy_n() but using CUDA arrays.
     */
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]);
    //! \brief Copy-convert complex-to-real.
    template<typename scalar_type>
    static void copy_n(std::complex<scalar_type> const source[], size_t num_entries, scalar_type destination[]){
        oneapi::convert(static_cast<long long>(num_entries), source, destination);
    }
    //! \brief Copy-convert real-to-complex.
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, std::complex<scalar_type> destination[]){
        oneapi::convert(static_cast<long long>(num_entries), source, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void scale(index num_entries, scalar_type data[], double scale_factor){
        oneapi::scale_data(num_entries, data, scale_factor);
    }
    /*!
     * \brief Complex by real scaling.
     */
    template<typename precision_type, typename index>
    static void scale(index num_entries, std::complex<precision_type> data[], double scale_factor){
        scale<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

namespace backend{
    /*!
     * \ingroup heffteoneapi
     * \brief Type-tag for the cuFFT backend
     */
    struct onemkl{};

    /*!
     * \ingroup heffteoneapi
     * \brief Indicate that the cuFFT backend has been enabled.
     */
    template<> struct is_enabled<onemkl> : std::true_type{};

    /*!
     * \ingroup heffteoneapi
     * \brief Defines the location type-tag and the cuda container.
     */
    template<>
    struct buffer_traits<onemkl>{
        //! \brief The oneMKL library uses data on the gpu device.
        using location = tag::gpu;
        //! \brief The data is managed by the cuda vector container.
        template<typename T> using container = heffte::gpu::vector<T>;
    };

    /*!
     * \ingroup heffteoneapi
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<onemkl>(){ return "onemkl"; }
}

/*!
 * \ingroup heffteoneapi
 * \brief Helper struct that defines the types and creates instances of one-dimensional executors.
 *
 * The struct is specialized for each backend.
 * Note that the MKL and oneMKL backends use identical executors where oneMKL decides how to handle
 * the device data in the backend.
 */
template<> struct one_dim_backend<backend::onemkl>{
    //! \brief Defines the complex-to-complex executor.
    using type = mkl_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = mkl_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    template<typename index>
    static std::unique_ptr<mkl_executor> make(box3d<index> const box, int dimension){
        return std::unique_ptr<mkl_executor>(new mkl_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    template<typename index>
    static std::unique_ptr<mkl_executor_r2c> make_r2c(box3d<index> const box, int dimension){
        return std::unique_ptr<mkl_executor_r2c>(new mkl_executor_r2c(box, dimension));
    }
};

namespace oneapi { // packer logic

/*!
 * \ingroup heffteoneapi
 * \brief Performs a direct-pack operation for data sitting on the GPU device.
 *
 * Launches a SYCL/DPC++ kernel.
 */
template<typename scalar_type, typename index>
void direct_pack(index nfast, index nmid, index nslow, index line_stride, index plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup heffteoneapi
 * \brief Performs a direct-unpack operation for data sitting on the GPU device.
 *
 * Launches a SYCL/DPC++ kernel.
 */
template<typename scalar_type, typename index>
void direct_unpack(index nfast, index nmid, index nslow, index line_stride, index plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup heffteoneapi
 * \brief Performs a transpose-unpack operation for data sitting on the GPU device.
 *
 * Launches a SYCL/DPC++ kernel.
 */
template<typename scalar_type, typename index>
void transpose_unpack(index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                      index buff_line_stride, index buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]);

}

/*!
 * \ingroup hefftepacking
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type, typename index>
    void pack(pack_plan_3d<index> const &plan, scalar_type const data[], scalar_type buffer[]) const{
        oneapi::direct_pack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, data, buffer);
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type, typename index>
    void unpack(pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        oneapi::direct_unpack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, buffer, data);
    }
};

/*!
 * \ingroup hefftepacking
 * \brief GPU version of the transpose packer.
 */
template<> struct transpose_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type, typename index>
    void pack(pack_plan_3d<index> const &plan, scalar_type const data[], scalar_type buffer[]) const{
        direct_packer<tag::gpu>().pack(plan, data, buffer); // packing is done the same way as the direct_packer
    }
    //! \brief Execute the planned transpose-unpack operation.
    template<typename scalar_type, typename index>
    void unpack(pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        oneapi::transpose_unpack<scalar_type>(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride,
                                            plan.buff_line_stride, plan.buff_plane_stride, plan.map[0], plan.map[1], plan.map[2], buffer, data);
    }
};

/*!
 * \ingroup heffteoneapi
 * \brief Specialization for the CPU case.
 */
template<> struct data_scaling<tag::gpu>{
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void apply(index num_entries, scalar_type *data, double scale_factor){
        oneapi::scale_data(static_cast<long long>(num_entries), data, scale_factor);
    }
    /*!
     * \brief Complex by real scaling.
     */
    template<typename precision_type, typename index>
    static void apply(index num_entries, std::complex<precision_type> *data, double scale_factor){
        apply<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

/*!
 * \ingroup heffteoneapi
 * \brief Sets the default options for the oneMKL backend.
 */
template<> struct default_plan_options<backend::onemkl>{
    //! \brief The reshape operations will not transpose the data.
    static const bool use_reorder = false;
};

}

#endif

#endif
