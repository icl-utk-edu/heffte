/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_ROCM_H
#define HEFFTE_BACKEND_ROCM_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_ROCM

#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <rocfft.h>
#include "heffte_backend_vector.h"

#ifdef Heffte_ENABLE_MAGMA
#include "heffte_magma_helpers.h"
#endif

/*!
 * \ingroup fft3d
 * \addtogroup heffterocm Backend rocfft
 *
 * Wrappers and template specializations related to the rocFFT backend.
 * Requires CMake option:
 * \code
 *  -D Heffte_ENABLE_ROCM=ON
 * \endcode
 *
 * In addition to the rocFFT wrappers, this also includes a series of kernels
 * for packing/unpacking/scaling the data, as well as a simple container
 * that wraps around ROCM arrays for RAII style of resource management.
 */

namespace heffte{

/*!
 * \ingroup heffterocm
 * \brief ROCM specific methods, vector-like container, error checking, etc.
 */
namespace rocm {
    /*!
     * \ingroup heffterocm
     * \brief Checks the status of a ROCm command and in case of a failure, converts it to a C++ exception.
     */
    inline void check_error(hipError_t status, std::string const &function_name){
        if (status != hipSuccess)
            throw std::runtime_error(function_name + " failed with message: " + std::to_string(status));
    }
    /*!
     * \ingroup heffterocm
     * \brief Checks the status of a cufft command and in case of a failure, converts it to a C++ exception.
     */
    inline void check_error(rocfft_status status, std::string const &function_name){
        if (status != rocfft_status_success)
            throw std::runtime_error(function_name + " failed with error code: " + std::to_string(status));
    }

    /*!
     * \ingroup heffterocm
     * \brief Convert real numbers to complex when both are located on the GPU device.
     *
     * Launches a ROCM kernel.
     */
    template<typename precision_type, typename index>
    void convert(hipStream_t stream, index num_entries, precision_type const source[], std::complex<precision_type> destination[]);
    /*!
     * \ingroup heffterocm
     * \brief Convert complex numbers to real when both are located on the GPU device.
     *
     * Launches a ROCM kernel.
     */
    template<typename precision_type, typename index>
    void convert(hipStream_t stream, index num_entries, std::complex<precision_type> const source[], precision_type destination[]);

    /*!
     * \ingroup heffterocm
     * \brief Scales real data (double or float) by the scaling factor.
     */
    template<typename scalar_type, typename index>
    void scale_data(hipStream_t stream, index num_entries, scalar_type *data, double scale_factor);

    /*!
     * \ingroup heffterocm
     * \brief Performs a direct-pack operation for data sitting on the GPU device.
     *
     * Launches a HIP kernel.
     */
    template<typename scalar_type, typename index>
    void direct_pack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                    scalar_type const source[], scalar_type destination[]);
    /*!
     * \ingroup heffterocm
     * \brief Performs a direct-unpack operation for data sitting on the GPU device.
     *
     * Launches a HIP kernel.
     */
    template<typename scalar_type, typename index>
    void direct_unpack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                    scalar_type const source[], scalar_type destination[]);
    /*!
     * \ingroup heffterocm
     * \brief Performs a transpose-unpack operation for data sitting on the GPU device.
     *
     * Launches a HIP kernel.
     */
    template<typename scalar_type, typename index>
    void transpose_unpack(hipStream_t stream, index nfast, index nmid, index nslow, index line_stride, index plane_stide,
                        index buff_line_stride, index buff_plane_stride, int map0, int map1, int map2,
                        scalar_type const source[], scalar_type destination[]);
}

namespace backend{
    /*!
     * \ingroup heffterocm
     * \brief Indicate that the cuFFT backend has been enabled.
     */
    template<> struct is_enabled<rocfft> : std::true_type{};

    /*!
     * \ingroup heffterocm
     * \brief The ROCm backend uses a HIP stream.
     */
    template<>
    struct device_instance<rocfft>{
        //! \brief Constructor, sets up the stream.
        device_instance(hipStream_t new_stream = nullptr) : _stream(new_stream){}
        //! \brief Returns the nullptr.
        hipStream_t stream(){ return _stream; }
        //! \brief Returns the nullptr (const case).
        hipStream_t stream() const{ return _stream; }
        //! \brief Syncs the execution with the queue.
        void synchronize_device() const{ rocm::check_error(hipStreamSynchronize(_stream), "device sync"); }
        //! \brief The CUDA stream to be used in all operations.
        mutable hipStream_t _stream;
        //! \brief The type for the internal stream.
        using stream_type = hipStream_t;
    };

    /*!
     * \ingroup heffterocm
     * \brief In ROCm mode, the default GPU backend is rocfft.
     */
    template<> struct default_backend<tag::gpu>{
        //! \brief Set the rocfft tag.
        using type = rocfft;
    };

    /*!
     * \ingroup heffterocm
     * \brief Specialization for the data operations in ROCm mode.
     */
    template<> struct data_manipulator<tag::gpu> {
        //! \brief Defines the backend_device.
        using backend_device = backend::device_instance<typename default_backend<tag::gpu>::type>;
        //! \brief Allocate memory.
        template<typename scalar_type>
        static scalar_type* allocate(hipStream_t stream, size_t num_entries){
            void *new_data;
            if (stream != nullptr) rocm::check_error( hipStreamSynchronize(stream), "hipStreamSynchronize()");
            rocm::check_error(hipMalloc(&new_data, num_entries * sizeof(scalar_type)), "hipMalloc()");
            return reinterpret_cast<scalar_type*>(new_data);
        }
        //! \brief Free memory.
        template<typename scalar_type>
        static void free(hipStream_t stream, scalar_type *pntr){
            if (pntr == nullptr) return;
            if (stream != nullptr) rocm::check_error( hipStreamSynchronize(stream), "hipStreamSynchronize()");
            rocm::check_error(hipFree(pntr), "hipFree()");
        }
        //! \brief Equivalent to std::copy_n() but using CUDA arrays.
        template<typename scalar_type>
        static void copy_n(hipStream_t stream, scalar_type const source[], size_t num_entries, scalar_type destination[]){
            if (stream == nullptr)
                rocm::check_error(hipMemcpy(destination, source, num_entries * sizeof(scalar_type), hipMemcpyDeviceToDevice), "data_manipulator::copy_n()");
            else
                rocm::check_error(hipMemcpyAsync(destination, source, num_entries * sizeof(scalar_type), hipMemcpyDeviceToDevice, stream), "data_manipulator::copy_n()");
        }
        //! \brief Copy-convert complex-to-real.
        template<typename scalar_type>
        static void copy_n(hipStream_t stream, std::complex<scalar_type> const source[], size_t num_entries, scalar_type destination[]){
            rocm::convert(stream, static_cast<long long>(num_entries), source, destination);
        }
        //! \brief Copy-convert real-to-complex.
        template<typename scalar_type>
        static void copy_n(hipStream_t stream, scalar_type const source[], size_t num_entries, std::complex<scalar_type> destination[]){
            rocm::convert(stream, static_cast<long long>(num_entries), source, destination);
        }
        //! \brief Copy the date from the device to the host.
        template<typename scalar_type>
        static void copy_device_to_host(hipStream_t stream, scalar_type const source[], size_t num_entries, scalar_type destination[]){
            rocm::check_error(hipMemcpyAsync(destination, source, num_entries * sizeof(scalar_type), hipMemcpyDeviceToHost, stream),
                            "device_to_host (rocm)");
        }
        //! \brief Copy the date from the device to the device.
        template<typename scalar_type>
        static void copy_device_to_device(hipStream_t stream, scalar_type const source[], size_t num_entries, scalar_type destination[]){
                rocm::check_error(hipMemcpyAsync(destination, source, num_entries * sizeof(scalar_type), hipMemcpyDeviceToDevice, stream),
                                "device_to_device (rocm)");
        }
        //! \brief Copy the date from the host to the device.
        template<typename scalar_type>
        static void copy_host_to_device(hipStream_t stream, scalar_type const source[], size_t num_entries, scalar_type destination[]){
            rocm::check_error(hipMemcpyAsync(destination, source, num_entries * sizeof(scalar_type), hipMemcpyHostToDevice, stream),
                            "host_to_device (rocm)");
        }
    };

    /*!
     * \ingroup heffterocm
     * \brief Defines the location type-tag and the cuda container.
     */
    template<>
    struct buffer_traits<rocfft>{
        //! \brief The rocfft library uses data on the gpu device.
        using location = tag::gpu;
        //! \brief The data is managed by the ROCm vector container.
        template<typename T> using container = heffte::gpu::device_vector<T, data_manipulator<tag::gpu>>;
    };
}

/*!
 * \ingroup heffterocm
 * \brief Plan for the r2c single precision transform.
 *
 * Note, this is a base template and does not specialize,
 * the complex case is handled in a specialization.
 */
template<typename precision_type, direction dir>
struct plan_rocfft{
    /*!
     * \brief Constructor and initializer of the plan.
     *
     * \param stream the hipStream_t to use for the transform
     * \param dir is the direction (forward or backward) for the plan
     * \param size is the number of entries in a 1-D transform
     * \param batch is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param rdist is the distance between the first entries of consecutive real sequences
     * \param cdist is the distance between the first entries of consecutive complex sequences
     */
    plan_rocfft(size_t size, size_t batch, size_t stride, size_t rdist, size_t cdist){

        rocfft_plan_description desc = nullptr;
        rocm::check_error( rocfft_plan_description_create(&desc), "rocm plan create");

        rocm::check_error(
            rocfft_plan_description_set_data_layout(
                desc,
                (dir == direction::forward) ? rocfft_array_type_real : rocfft_array_type_hermitian_interleaved,
                (dir == direction::forward) ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_real,
                nullptr, nullptr,
                1, &stride, (dir == direction::forward) ? rdist : cdist,
                1, &stride, (dir == direction::forward) ? cdist : rdist
            ),
            "plan layout"
        );

        rocm::check_error(
        rocfft_plan_create(&plan, rocfft_placement_notinplace,
                           (dir == direction::forward) ? rocfft_transform_type_real_forward : rocfft_transform_type_real_inverse,
                           (std::is_same<precision_type, float>::value)? rocfft_precision_single : rocfft_precision_double,
                           1, &size, batch, desc),
        "plan create");

        rocm::check_error( rocfft_plan_get_work_buffer_size(plan, &worksize), "get_worksize");

        rocm::check_error( rocfft_plan_description_destroy(desc), "rocm plan destroy");
    }
    //! \brief Destructor, deletes the plan.
    ~plan_rocfft(){ rocfft_plan_destroy(plan); }
    //! \brief Custom conversion to the rocfft_plan.
    operator rocfft_plan() const{ return plan; }
    //! \brief Return the worksize.
    size_t size_work() const{ return worksize; }

private:
    //! \brief The cufft opaque structure (pointer to struct).
    rocfft_plan plan;
    //! \brief The size of the scratch workspace.
    size_t worksize;
};

/*!
 * \ingroup heffterocm
 * \brief Plan for the single precision complex transform.
 */
template<typename precision_type, direction dir>
struct plan_rocfft<std::complex<precision_type>, dir>{
    /*!
     * \brief Constructor, takes inputs identical to cufftMakePlanMany().
     *
     * \param size is the number of entries in a 1-D transform
     * \param batch is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_rocfft(size_t size, size_t batch, size_t stride, size_t dist) : plan(nullptr), worksize(0){
        rocfft_plan_description desc = nullptr;
        rocm::check_error( rocfft_plan_description_create(&desc), "rocm plan create");

        rocm::check_error(
            rocfft_plan_description_set_data_layout(
                desc,
                rocfft_array_type_complex_interleaved,
                rocfft_array_type_complex_interleaved,
                nullptr, nullptr,
                1, &stride, dist, 1, &stride, dist
            ),
            "plan layout"
        );

        rocm::check_error(
        rocfft_plan_create(&plan, rocfft_placement_inplace,
                           (dir == direction::forward) ? rocfft_transform_type_complex_forward : rocfft_transform_type_complex_inverse,
                           (std::is_same<precision_type, float>::value)? rocfft_precision_single : rocfft_precision_double,
                           1, &size, batch, desc),
        "plan create");

        rocm::check_error( rocfft_plan_get_work_buffer_size(plan, &worksize), "get_worksize");

        rocm::check_error( rocfft_plan_description_destroy(desc), "rocm plan destroy");
    }
    //! \brief Destructor, deletes the plan.
    ~plan_rocfft(){ rocm::check_error( rocfft_plan_destroy(plan), "plan destory"); }
    //! \brief Custom conversion to the rocfft_plan.
    operator rocfft_plan() const{ return plan; }
    //! \brief Return the worksize.
    size_t size_work() const{ return worksize; }

private:
    //! \brief The rocfft opaque structure (pointer to struct).
    rocfft_plan plan;
    size_t worksize;
};

/*!
 * \ingroup heffterocm
 * \brief Wrapper around the rocFFT API.
 *
 * A single class that manages the plans and executions of rocFFT
 * so that a single API is provided for all backends.
 * The executor operates on a box and performs 1-D FFTs
 * for the given dimension.
 * The class silently manages the plans and buffers needed
 * for the different types.
 * All input and output arrays must have size equal to the box.
 */
class rocfft_executor{
public:
    //! \brief Constructor, specifies the box and dimension.
    template<typename index>
    rocfft_executor(hipStream_t active_stream, box3d<index> const box, int dimension) :
        stream(active_stream),
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == box.order[0]) ? size : 1),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        block_stride(box.osize(0) * box.osize(1)),
        total_size(box.count())
    {}

    //! \brief Perform an in-place FFT on the data in the given direction.
    template<typename precision_type, direction dir>
    void execute(std::complex<precision_type> data[]) const{
        if (std::is_same<precision_type, float>::value){
            if (dir == direction::forward)
                make_plan(ccomplex_forward);
            else
                make_plan(ccomplex_backward);
        }else{
            if (dir == direction::forward)
                make_plan(zcomplex_forward);
            else
                make_plan(zcomplex_backward);
        }
        rocfft_execution_info info;
        rocfft_execution_info_create(&info);
        rocfft_execution_info_set_stream(info, stream);

        size_t wsize = (std::is_same<precision_type, float>::value) ?
                            ((dir == direction::forward) ? ccomplex_forward->size_work() : ccomplex_backward->size_work()) :
                            ((dir == direction::forward) ? zcomplex_forward->size_work() : zcomplex_backward->size_work());
        backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>>(stream, wsize);
            rocfft_execution_info_set_work_buffer(info, reinterpret_cast<void*>(work_buff.data()), wsize);
        }

        for(int i=0; i<blocks; i++){
            void* block_data = reinterpret_cast<void*>(data + i * block_stride);
            rocm::check_error( rocfft_execute(
                (std::is_same<precision_type, float>::value) ?
                    ((dir == direction::forward) ? *ccomplex_forward : *ccomplex_backward) :
                    ((dir == direction::forward) ? *zcomplex_forward : *zcomplex_backward),
                &block_data, nullptr, info), "rocfft execute");
        }
        rocfft_execution_info_destroy(info);
    }

    //! \brief Forward fft, float-complex case.
    template<typename precision_type>
    void forward(std::complex<precision_type> data[]) const{
        execute<precision_type, direction::forward>(data);
    }
    //! \brief Backward fft, float-complex case.
    template<typename precision_type>
    void backward(std::complex<precision_type> data[]) const{
        execute<precision_type, direction::backward>(data);
    }

    //! \brief Converts the deal data to complex and performs float-complex forward transform.
    template<typename precision_type>
    void forward(precision_type const indata[], std::complex<precision_type> outdata[]) const{
        rocm::convert(stream, total_size, indata, outdata);
        forward(outdata);
    }
    //! \brief Performs backward float-complex transform and truncates the complex part of the result.
    template<typename precision_type>
    void backward(std::complex<precision_type> indata[], precision_type outdata[]) const{
        backward(indata);
        rocm::convert(stream, total_size, indata, outdata);
    }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    //! \brief Helper template to create the plan.
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_rocfft<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_rocfft<scalar_type, dir>>(new plan_rocfft<scalar_type, dir>(size, howmanyffts, stride, dist));
    }

    mutable hipStream_t stream;

    int size, howmanyffts, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_rocfft<std::complex<float>, direction::forward>> ccomplex_forward;
    mutable std::unique_ptr<plan_rocfft<std::complex<float>, direction::backward>> ccomplex_backward;
    mutable std::unique_ptr<plan_rocfft<std::complex<double>, direction::forward>> zcomplex_forward;
    mutable std::unique_ptr<plan_rocfft<std::complex<double>, direction::backward>> zcomplex_backward;
};

/*!
 * \ingroup heffterocm
 * \brief Wrapper to cuFFT API for real-to-complex transform with shortening of the data.
 *
 * Serves the same purpose of heffte::cufft_executor but only real input is accepted
 * and only the unique (non-conjugate) coefficients are computed.
 * All real arrays must have size of real_size() and all complex arrays must have size complex_size().
 */
class rocfft_executor_r2c{
public:
    /*!
     * \brief Constructor defines the box and the dimension of reduction.
     *
     * Note that the result sits in the box returned by box.r2c(dimension).
     */
    template<typename index>
    rocfft_executor_r2c(hipStream_t active_stream, box3d<index> const box, int dimension) :
        stream(active_stream),
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        rdist((dimension == box.order[0]) ? size : 1),
        cdist((dimension == box.order[0]) ? size/2 + 1 : 1),
        rblock_stride(box.osize(0) * box.osize(1)),
        cblock_stride(box.osize(0) * (box.osize(1)/2 + 1)),
        rsize(box.count()),
        csize(box.r2c(dimension).count())
    {}

    //! \brief Forward transform, single precision.
    template<typename precision_type>
    void forward(precision_type const indata[], std::complex<precision_type> outdata[]) const{
        if (std::is_same<precision_type, float>::value){
            make_plan(sforward);
        }else{
            make_plan(dforward);
        }

        rocfft_execution_info info;
        rocfft_execution_info_create(&info);
        rocfft_execution_info_set_stream(info, stream);

        size_t wsize = (std::is_same<precision_type, float>::value) ? sforward->size_work() : dforward->size_work();
        backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>>(stream, wsize);
            rocfft_execution_info_set_work_buffer(info, reinterpret_cast<void*>(work_buff.data()), wsize);
        }
        backend::buffer_traits<backend::rocfft>::container<precision_type> copy_indata(stream, indata, indata + real_size());

        for(int i=0; i<blocks; i++){
            void *rdata = const_cast<void*>(reinterpret_cast<void const*>(copy_indata.data() + i * rblock_stride));
            void *cdata = reinterpret_cast<void*>(outdata + i * cblock_stride);
            rocm::check_error( rocfft_execute(
                (std::is_same<precision_type, float>::value) ? *sforward : *dforward,
                &rdata, &cdata, info), "rocfft execute");
        }
        rocfft_execution_info_destroy(info);
    }
    //! \brief Backward transform, single precision.
    template<typename precision_type>
    void backward(std::complex<precision_type> const indata[], precision_type outdata[]) const{
        if (std::is_same<precision_type, float>::value){
            make_plan(sbackward);
        }else{
            make_plan(dbackward);
        }

        rocfft_execution_info info;
        rocfft_execution_info_create(&info);

        size_t wsize = (std::is_same<precision_type, float>::value) ? sbackward->size_work() : dbackward->size_work();
        backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>>(stream, wsize);
            rocfft_execution_info_set_work_buffer(info, reinterpret_cast<void*>(work_buff.data()), wsize);
        }
        backend::buffer_traits<backend::rocfft>::container<std::complex<precision_type>>
                copy_indata(stream, indata, indata + complex_size());

        for(int i=0; i<blocks; i++){
            void *cdata = const_cast<void*>(reinterpret_cast<void const*>(copy_indata.data() + i * cblock_stride));
            void *rdata = reinterpret_cast<void*>(outdata + i * rblock_stride);
            rocm::check_error( rocfft_execute(
                (std::is_same<precision_type, float>::value) ? *sbackward : *dbackward,
                &cdata, &rdata, info), "rocfft execute");
        }
        rocfft_execution_info_destroy(info);
    }

    //! \brief Returns the size of the box with real data.
    int real_size() const{ return rsize; }
    //! \brief Returns the size of the box with complex coefficients.
    int complex_size() const{ return csize; }

private:
    //! \brief Helper template to initialize the plan.
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_rocfft<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_rocfft<scalar_type, dir>>(new plan_rocfft<scalar_type, dir>(size, howmanyffts, stride, rdist, cdist));
    }

    mutable hipStream_t stream;

    int size, howmanyffts, stride, blocks;
    int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_rocfft<float, direction::forward>> sforward;
    mutable std::unique_ptr<plan_rocfft<double, direction::forward>> dforward;
    mutable std::unique_ptr<plan_rocfft<float, direction::backward>> sbackward;
    mutable std::unique_ptr<plan_rocfft<double, direction::backward>> dbackward;
};

/*!
 * \ingroup heffterocm
 * \brief Helper struct that defines the types and creates instances of one-dimensional executors.
 *
 * The struct is specialized for each backend.
 */
template<> struct one_dim_backend<backend::rocfft>{
    //! \brief Defines the complex-to-complex executor.
    using type = rocfft_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = rocfft_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    template<typename index>
    static std::unique_ptr<rocfft_executor> make(hipStream_t stream, box3d<index> const box, int dimension){
        return std::unique_ptr<rocfft_executor>(new rocfft_executor(stream, box, dimension));
    }
    //! \brief Constructs a 2D executor from two 1D ones.
    template<typename index>
    static std::unique_ptr<rocfft_executor> make(hipStream_t stream,  box3d<index> const&, int, int){
        throw std::runtime_error("2d rocfft executor not implemented");
        return std::unique_ptr<rocfft_executor>();
    }
    //! \brief Returns true if the transforms in the two directions can be merged into one.
    template<typename index> static bool can_merge(box3d<index> const&, int, int){ return false; }
    //! \brief Constructs a real-to-complex executor.
    template<typename index>
    static std::unique_ptr<rocfft_executor_r2c> make_r2c(hipStream_t stream, box3d<index> const box, int dimension){
        return std::unique_ptr<rocfft_executor_r2c>(new rocfft_executor_r2c(stream, box, dimension));
    }
};

/*!
 * \ingroup hefftepacking
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type, typename index>
    void pack(hipStream_t stream, pack_plan_3d<index> const &plan, scalar_type const data[], scalar_type buffer[]) const{
        rocm::direct_pack(stream, plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, data, buffer);
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type, typename index>
    void unpack(hipStream_t stream, pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        rocm::direct_unpack(stream, plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, buffer, data);
    }
};

/*!
 * \ingroup hefftepacking
 * \brief GPU version of the transpose packer.
 */
template<> struct transpose_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type, typename index>
    void pack(hipStream_t stream, pack_plan_3d<index> const &plan, scalar_type const data[], scalar_type buffer[]) const{
        direct_packer<tag::gpu>().pack(stream, plan, data, buffer); // packing is done the same way as the direct_packer
    }
    //! \brief Execute the planned transpose-unpack operation.
    template<typename scalar_type, typename index>
    void unpack(hipStream_t stream, pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        rocm::transpose_unpack<scalar_type>(stream, plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride,
                                            plan.buff_line_stride, plan.buff_plane_stride, plan.map[0], plan.map[1], plan.map[2], buffer, data);
    }
};

namespace data_scaling {
    /*!
     * \ingroup heffterocm
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void apply(hipStream_t stream, index num_entries, scalar_type *data, double scale_factor){
        rocm::scale_data<scalar_type, long long>(stream, static_cast<long long>(num_entries), data, scale_factor);
    }
    /*!
     * \ingroup heffterocm
     * \brief Complex by real scaling.
     */
    template<typename precision_type, typename index>
    static void apply(hipStream_t stream, index num_entries, std::complex<precision_type> *data, double scale_factor){
        apply<precision_type>(stream, 2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

/*!
 * \ingroup heffterocm
 * \brief Sets the default options for the cufft backend.
 */
template<> struct default_plan_options<backend::rocfft>{
    //! \brief The reshape operations will not transpose the data.
    static const bool use_reorder = true;
};

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
