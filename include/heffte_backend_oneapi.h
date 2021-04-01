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
#include "oneapi/mkl/dfti.hpp"

#ifdef Heffte_ENABLE_MAGMA
// will enable once MAGMA has a DPC++/SYCL backend
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
 *
 * The name is chosen distinct from the oneMKL name that use "oneapi".
 */
namespace oapi {
    //! \brief Creates a new SYCL queue.
    sycl::queue* make_sycl_queue();

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

    /*!
     * \ingroup heffteoneapi
     * \brief There is an instance of an internal sycl::queue, used mostly for testing.
     */
    struct heffte_internal_sycl_queue{
        //! \brief Constructor, creates a default queue.
        heffte_internal_sycl_queue(){
            queue_ptr = std::unique_ptr<sycl::queue>(make_sycl_queue());
        }
        //! \brief Returns a reference to the queue.
        operator sycl::queue& () { return *queue_ptr.get(); }
        //! \brief The queue is accessible through the -> operator.
        sycl::queue* operator ->() { return queue_ptr.get(); }
        //! \brief Sync the queue.
        void wait(){ queue_ptr->wait(); }

        //! \brief Container holding the queue.
        std::unique_ptr<sycl::queue> queue_ptr;
    };

    extern heffte_internal_sycl_queue def_queue;
}

namespace gpu {
    /*!
     * \ingroup heffteoneapi
     * \brief Device vector for the oneAPI backends.
     */
    template<typename scalar_type>
    using vector = device_vector<scalar_type, oapi::memory_manager>;

    /*!
     * \ingroup heffteoneapi
     * \brief Transfer helpers for the oneAPI backends.
     */
    using transfer = device_transfer<oapi::memory_manager>;

};

/*!
 * \ingroup heffteoneapi
 * \brief Cuda specific methods, vector-like container, error checking, etc.
 */
namespace oapi {

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
        oapi::convert(static_cast<long long>(num_entries), source, destination);
    }
    //! \brief Copy-convert real-to-complex.
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, std::complex<scalar_type> destination[]){
        oapi::convert(static_cast<long long>(num_entries), source, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void scale(index num_entries, scalar_type data[], double scale_factor){
        oapi::scale_data(num_entries, data, scale_factor);
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

    /*!
     * \ingroup heffteoneapi
     * \brief Specialization that contains the sycl::queue needed for the DPC++ backend.
     */
    template<>
    struct auxiliary_variables<onemkl>{
        //! \brief Empty constructor.
        auxiliary_variables() : queue_container(heffte::oapi::make_sycl_queue()){}
        //! \brief Default destructor.
        virtual ~auxiliary_variables() = default;
        //! \brief Returns the nullptr.
        sycl::queue* gpu_queue(){ return queue_container.get(); }
        //! \brief The sycl::queue, either user provided or created by heFFTe.
        std::unique_ptr<sycl::queue> queue_container;
    };
}

/*!
 * \ingroup heffteoneapi
 * \brief Wrapper around the oneMKL API.
 *
 * A single class that manages the plans and executions of oneMKL FFTs.
 * Handles the complex-to-complex cases.
 */
class onemkl_executor{
public:
    //! \brief Constructor, specifies the box and dimension.
    template<typename index>
    onemkl_executor(sycl::queue *inq, box3d<index> const box, int dimension) :
        q(inq),
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == box.order[0]) ? size : 1),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        block_stride(box.osize(0) * box.osize(1)),
        total_size(box.count()),
        init_cplan(false), init_zplan(false),
        cplan(size), zplan(size)
    {}

    //! \brief Forward fft, float-complex case.
    void forward(std::complex<float> data[]) const{
        if (not init_cplan) make_plan(cplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_forward(cplan, data + i * block_stride);
        q->wait();
    }
    //! \brief Backward fft, float-complex case.
    void backward(std::complex<float> data[]) const{
        if (not init_cplan) make_plan(cplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_backward(cplan, data + i * block_stride);
        q->wait();
    }
    //! \brief Forward fft, double-complex case.
    void forward(std::complex<double> data[]) const{
        if (not init_zplan) make_plan(zplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_forward(zplan, data + i * block_stride);
        q->wait();
    }
    //! \brief Backward fft, double-complex case.
    void backward(std::complex<double> data[]) const{
        if (not init_zplan) make_plan(zplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_backward(zplan, data + i * block_stride);
        q->wait();
    }

    //! \brief Converts the deal data to complex and performs float-complex forward transform.
    void forward(float const indata[], std::complex<float> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<float>(indata[i]);
        forward(outdata);
    }
    //! \brief Performs backward float-complex transform and truncates the complex part of the result.
    void backward(std::complex<float> indata[], float outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }
    //! \brief Converts the deal data to complex and performs double-complex forward transform.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<double>(indata[i]);
        forward(outdata);
    }
    //! \brief Performs backward double-complex transform and truncates the complex part of the result.
    void backward(std::complex<double> indata[], double outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    //! \brief Helper template to create the plan.
    template<typename onemkl_plan_type>
    void make_plan(onemkl_plan_type &plan) const{
        plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts);
        plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_INPLACE);
        MKL_LONG slstride[] = {0, static_cast<MKL_LONG>(stride)};
        plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, slstride);
        plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, slstride);
        plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, (MKL_LONG) dist);
        plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, (MKL_LONG) dist);
        plan.commit(*q);
        q->wait();

        if (std::is_same<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>, onemkl_plan_type>::value)
            init_cplan = true;
        else
            init_zplan = true;
    }

    sycl::queue *q;
    int size, howmanyffts, stride, dist, blocks, block_stride, total_size;

    mutable bool init_cplan, init_zplan;
    mutable oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX> cplan;
    mutable oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX> zplan;
};

/*!
 * \ingroup heffteoneapi
 * \brief Wrapper to oneMKL API for real-to-complex transform with shortening of the data.
 *
 * Serves the same purpose of heffte::onemkl_executor but only real input is accepted
 * and only the unique (non-conjugate) coefficients are computed.
 * All real arrays must have size of real_size() and all complex arrays must have size complex_size().
 */
class onemkl_executor_r2c{
public:
    /*!
     * \brief Constructor defines the box and the dimension of reduction.
     *
     * Note that the result sits in the box returned by box.r2c(dimension).
     */
    template<typename index>
    onemkl_executor_r2c(sycl::queue *inq, box3d<index> const box, int dimension) :
        q(inq),
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        rdist((dimension == box.order[0]) ? size : 1),
        cdist((dimension == box.order[0]) ? size/2 + 1 : 1),
        rblock_stride(box.osize(0) * box.osize(1)),
        cblock_stride(box.osize(0) * (box.osize(1)/2 + 1)),
        rsize(box.count()),
        csize(box.r2c(dimension).count()),
        init_splan(false), init_dplan(false),
        splan(size), dplan(size)
    {}

    //! \brief Forward transform, single precision.
    void forward(float const indata[], std::complex<float> outdata[]) const{
        if (not init_splan) make_plan(splan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_forward(splan, const_cast<float*>(indata + i * rblock_stride), reinterpret_cast<float*>(outdata + i * cblock_stride));
        q->wait();
    }
    //! \brief Backward transform, single precision.
    void backward(std::complex<float> const indata[], float outdata[]) const{
        if (not init_splan) make_plan(splan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_backward(splan, reinterpret_cast<float*>(const_cast<std::complex<float>*>(indata + i * cblock_stride)), outdata + i * rblock_stride);
        q->wait();
    }
    //! \brief Forward transform, double precision.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        if (not init_dplan) make_plan(dplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_forward(dplan, const_cast<double*>(indata + i * rblock_stride), reinterpret_cast<double*>(outdata + i * cblock_stride));
        q->wait();
    }
    //! \brief Backward transform, double precision.
    void backward(std::complex<double> const indata[], double outdata[]) const{
        if (not init_dplan) make_plan(dplan);
        for(int i=0; i<blocks; i++)
            oneapi::mkl::dft::compute_backward(dplan, reinterpret_cast<double*>(const_cast<std::complex<double>*>(indata + i * cblock_stride)), outdata + i * rblock_stride);
        q->wait();
    }

    //! \brief Returns the size of the box with real data.
    int real_size() const{ return rsize; }
    //! \brief Returns the size of the box with complex coefficients.
    int complex_size() const{ return csize; }

private:
    //! \brief Helper template to initialize the plan.
    template<typename onemkl_plan_type>
    void make_plan(onemkl_plan_type &plan) const{
        plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts);
        plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
        plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        MKL_LONG slstride[] = {0, static_cast<MKL_LONG>(stride)};
        plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, slstride);
        plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, slstride);
        plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, (MKL_LONG) rdist);
        plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, (MKL_LONG) cdist);
        plan.commit(*q);

        if (std::is_same<oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>, onemkl_plan_type>::value)
            init_splan = true;
        else
            init_dplan = true;
        q->wait();
    }

    sycl::queue *q;

    int size, howmanyffts, stride, blocks;
    int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable bool init_splan, init_dplan;
    mutable oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL> splan;
    mutable oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL> dplan;
};

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
    using type = onemkl_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = onemkl_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    template<typename index>
    static std::unique_ptr<onemkl_executor> make(sycl::queue *q, box3d<index> const box, int dimension){
        return (q != nullptr) ?
            std::unique_ptr<onemkl_executor>(new onemkl_executor(q, box, dimension)) :
            std::unique_ptr<onemkl_executor>(new onemkl_executor(oapi::def_queue.queue_ptr.get(), box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    template<typename index>
    static std::unique_ptr<onemkl_executor_r2c> make_r2c(sycl::queue *q, box3d<index> const box, int dimension){
        return (q != nullptr) ?
            std::unique_ptr<onemkl_executor_r2c>(new onemkl_executor_r2c(q, box, dimension)) :
            std::unique_ptr<onemkl_executor_r2c>(new onemkl_executor_r2c(oapi::def_queue.queue_ptr.get(), box, dimension));
    }
};

namespace oapi { // packer logic

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
        oapi::direct_pack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, data, buffer);
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type, typename index>
    void unpack(pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        oapi::direct_unpack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, buffer, data);
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
        oapi::transpose_unpack<scalar_type>(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride,
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
        oapi::scale_data(static_cast<long long>(num_entries), data, scale_factor);
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
