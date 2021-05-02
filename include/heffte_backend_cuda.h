/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_CUDA_H
#define HEFFTE_BACKEND_CUDA_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_CUDA

#include <cufft.h>
#include "heffte_backend_vector.h"

#ifdef Heffte_ENABLE_MAGMA
#include "heffte_magma_helpers.h"
#endif

/*!
 * \ingroup fft3d
 * \addtogroup hefftecuda Backend cufft
 *
 * Wrappers and template specializations related to the cuFFT backend.
 * Requires CMake option:
 * \code
 *  -D Heffte_ENABLE_CUDA=ON
 * \endcode
 *
 * In addition to the cuFFT wrappers, this also includes a series of kernels
 * for packing/unpacking/scaling the data, as well as a simple container
 * that wraps around CUDA arrays for RAII style of resource management.
 */

namespace heffte{

/*!
 * \ingroup hefftecuda
 * \brief CUDA specific methods, vector-like container, error checking, etc.
 */
namespace cuda {
    /*!
     * \ingroup hefftecuda
     * \brief Memory management operation specific to CUDA, see gpu::device_vector.
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
     * \ingroup hefftecuda
     * \brief Device vector for the CUDA backends.
     */
    template<typename scalar_type>
    using vector = device_vector<scalar_type, cuda::memory_manager>;

    /*!
     * \ingroup hefftecuda
     * \brief Transfer helpers for the CUDA backends.
     */
    using transfer = device_transfer<cuda::memory_manager>;

};

/*!
 * \ingroup hefftecuda
 * \brief Cuda specific methods, vector-like container, error checking, etc.
 */
namespace cuda {

    /*!
     * \ingroup hefftecuda
     * \brief Checks the status of a CUDA command and in case of a failure, converts it to a C++ exception.
     */
    void check_error(cudaError_t status, std::string const &function_name);
    /*!
     * \ingroup hefftecuda
     * \brief Checks the status of a cufft command and in case of a failure, converts it to a C++ exception.
     */
    inline void check_error(cufftResult status, std::string const &function_name){
        if (status != CUFFT_SUCCESS)
            throw std::runtime_error(function_name + " failed with error code: " + std::to_string(status));
    }

    /*!
     * \ingroup hefftecuda
     * \brief Convert real numbers to complex when both are located on the GPU device.
     *
     * Launches a CUDA kernel.
     */
    template<typename precision_type, typename index>
    void convert(index num_entries, precision_type const source[], std::complex<precision_type> destination[]);
    /*!
     * \ingroup hefftecuda
     * \brief Convert complex numbers to real when both are located on the GPU device.
     *
     * Launches a CUDA kernel.
     */
    template<typename precision_type, typename index>
    void convert(index num_entries, std::complex<precision_type> const source[], precision_type destination[]);

    /*!
     * \ingroup hefftecuda
     * \brief Scales real data (double or float) by the scaling factor.
     */
    template<typename scalar_type, typename index>
    void scale_data(index num_entries, scalar_type *data, double scale_factor);
}

/*!
 * \ingroup hefftecuda
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
        cuda::convert(static_cast<long long>(num_entries), source, destination);
    }
    //! \brief Copy-convert real-to-complex.
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, std::complex<scalar_type> destination[]){
        cuda::convert(static_cast<long long>(num_entries), source, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void scale(index num_entries, scalar_type data[], double scale_factor){
        cuda::scale_data(num_entries, data, scale_factor);
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
     * \ingroup hefftecuda
     * \brief Type-tag for the cuFFT backend
     */
    struct cufft{};

    /*!
     * \ingroup hefftecuda
     * \brief Indicate that the cuFFT backend has been enabled.
     */
    template<> struct is_enabled<cufft> : std::true_type{};

    /*!
     * \ingroup hefftecuda
     * \brief Defines the location type-tag and the cuda container.
     */
    template<>
    struct buffer_traits<cufft>{
        //! \brief The cufft library uses data on the gpu device.
        using location = tag::gpu;
        //! \brief The data is managed by the cuda vector container.
        template<typename T> using container = heffte::gpu::vector<T>;
    };

    /*!
     * \ingroup hefftecuda
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<cufft>(){ return "cufft"; }
}

/*!
 * \ingroup hefftecuda
 * \brief Recognize the cuFFT single precision complex type.
 */
template<> struct is_ccomplex<cufftComplex> : std::true_type{};
/*!
 * \ingroup hefftecuda
 * \brief Recognize the cuFFT double precision complex type.
 */
template<> struct is_zcomplex<cufftDoubleComplex> : std::true_type{};

/*!
 * \ingroup hefftecuda
 * \brief Base plan for cufft, using only the specialization for float and double complex.
 *
 * Similar to heffte::plan_fftw but applies to the cufft backend.
 */
template<typename> struct plan_cufft{};

/*!
 * \ingroup hefftecuda
 * \brief Plan for the single precision complex transform.
 */
template<> struct plan_cufft<std::complex<float>>{
    /*!
     * \brief Constructor, takes inputs identical to cufftMakePlanMany().
     *
     * \param size is the number of entries in a 1-D transform
     * \param batch is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_cufft(int size, int batch, int stride, int dist){
        size_t work_size = 0;
        cuda::check_error(cufftCreate(&plan), "plan_cufft<std::complex<float>>::cufftCreate()");
        cuda::check_error(
            cufftMakePlanMany(plan, 1, &size, &size, stride, dist, &size, stride, dist, CUFFT_C2C, batch, &work_size),
            "plan_cufft<std::complex<float>>::cufftMakePlanMany()"
        );
    }
    //! \brief Destructor, deletes the plan.
    ~plan_cufft(){ cufftDestroy(plan); }
    //! \brief Custom conversion to the cufftHandle.
    operator cufftHandle() const{ return plan; }

private:
    //! \brief The cufft opaque structure (pointer to struct).
    cufftHandle plan;
};
/*!
 * \ingroup hefftecuda
 * \brief Specialization for double complex.
 */
template<> struct plan_cufft<std::complex<double>>{
    //! \brief Identical to the float-complex specialization.
    plan_cufft(int size, int batch, int stride, int dist){
        size_t work_size = 0;
        cuda::check_error(cufftCreate(&plan), "plan_cufft<std::complex<double>>::cufftCreate()");
        cuda::check_error(
            cufftMakePlanMany(plan, 1, &size, &size, stride, dist, &size, stride, dist, CUFFT_Z2Z, batch, &work_size),
            "plan_cufft<std::complex<double>>::cufftMakePlanMany()"
        );
    }
    //! \brief Identical to the float-complex specialization.
    ~plan_cufft(){ cufftDestroy(plan); }
    //! \brief Identical to the float-complex specialization.
    operator cufftHandle() const{ return plan; }

private:
    //! \brief Identical to the float-complex specialization.
    cufftHandle plan;
};

/*!
 * \ingroup hefftecuda
 * \brief Wrapper around the cuFFT API.
 *
 * A single class that manages the plans and executions of cuFFT
 * so that a single API is provided for all backends.
 * The executor operates on a box and performs 1-D FFTs
 * for the given dimension.
 * The class silently manages the plans and buffers needed
 * for the different types.
 * All input and output arrays must have size equal to the box.
 */
class cufft_executor{
public:
    //! \brief Constructor, specifies the box and dimension.
    template<typename index>
    cufft_executor(box3d<index> const box, int dimension) :
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == box.order[0]) ? size : 1),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        block_stride(box.osize(0) * box.osize(1)),
        total_size(box.count())
    {}

    //! \brief Forward fft, float-complex case.
    void forward(std::complex<float> data[]) const{
        make_plan(ccomplex_plan);
        for(int i=0; i<blocks; i++){
            cufftComplex* block_data = reinterpret_cast<cufftComplex*>(data + i * block_stride);
            cuda::check_error(cufftExecC2C(*ccomplex_plan, block_data, block_data, CUFFT_FORWARD), "cufft_executor::cufftExecC2C() forward");
        }
    }
    //! \brief Backward fft, float-complex case.
    void backward(std::complex<float> data[]) const{
        make_plan(ccomplex_plan);
        for(int i=0; i<blocks; i++){
            cufftComplex* block_data = reinterpret_cast<cufftComplex*>(data + i * block_stride);
            cuda::check_error(cufftExecC2C(*ccomplex_plan, block_data, block_data, CUFFT_INVERSE), "cufft_executor::cufftExecC2C() backward");
        }
    }
    //! \brief Forward fft, double-complex case.
    void forward(std::complex<double> data[]) const{
        make_plan(zcomplex_plan);
        for(int i=0; i<blocks; i++){
            cufftDoubleComplex* block_data = reinterpret_cast<cufftDoubleComplex*>(data + i * block_stride);
            cuda::check_error(cufftExecZ2Z(*zcomplex_plan, block_data, block_data, CUFFT_FORWARD), "cufft_executor::cufftExecZ2Z() forward");
        }
    }
    //! \brief Backward fft, double-complex case.
    void backward(std::complex<double> data[]) const{
        make_plan(zcomplex_plan);
        for(int i=0; i<blocks; i++){
            cufftDoubleComplex* block_data = reinterpret_cast<cufftDoubleComplex*>(data + i * block_stride);
            cuda::check_error(cufftExecZ2Z(*zcomplex_plan, block_data, block_data, CUFFT_INVERSE), "cufft_executor::cufftExecZ2Z() backward");
        }
    }

    //! \brief Converts the deal data to complex and performs float-complex forward transform.
    void forward(float const indata[], std::complex<float> outdata[]) const{
        cuda::convert(total_size, indata, outdata);
        forward(outdata);
    }
    //! \brief Performs backward float-complex transform and truncates the complex part of the result.
    void backward(std::complex<float> indata[], float outdata[]) const{
        backward(indata);
        cuda::convert(total_size, indata, outdata);
    }
    //! \brief Converts the deal data to complex and performs double-complex forward transform.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        cuda::convert(total_size, indata, outdata);
        forward(outdata);
    }
    //! \brief Performs backward double-complex transform and truncates the complex part of the result.
    void backward(std::complex<double> indata[], double outdata[]) const{
        backward(indata);
        cuda::convert(total_size, indata, outdata);
    }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    //! \brief Helper template to create the plan.
    template<typename scalar_type>
    void make_plan(std::unique_ptr<plan_cufft<scalar_type>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_cufft<scalar_type>>(new plan_cufft<scalar_type>(size, howmanyffts, stride, dist));
    }

    int size, howmanyffts, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_cufft<std::complex<float>>> ccomplex_plan;
    mutable std::unique_ptr<plan_cufft<std::complex<double>>> zcomplex_plan;
};

/*!
 * \ingroup hefftecuda
 * \brief Plan for the r2c single precision transform.
 */
template<> struct plan_cufft<float>{
    /*!
     * \brief Constructor, takes inputs identical to cufftMakePlanMany().
     *
     * \param dir is the direction (forward or backward) for the plan
     * \param size is the number of entries in a 1-D transform
     * \param batch is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param rdist is the distance between the first entries of consecutive real sequences
     * \param cdist is the distance between the first entries of consecutive complex sequences
     */
    plan_cufft(direction dir, int size, int batch, int stride, int rdist, int cdist){
        size_t work_size = 0;
        cuda::check_error(cufftCreate(&plan), "plan_cufft<float>::cufftCreate()");
        if (dir == direction::forward){
            cuda::check_error(
                cufftMakePlanMany(plan, 1, &size, &size, stride, rdist, &size, stride, cdist, CUFFT_R2C, batch, &work_size),
                "plan_cufft<float>::cufftMakePlanMany() (forward)"
            );
        }else{
            cuda::check_error(
                cufftMakePlanMany(plan, 1, &size, &size, stride, cdist, &size, stride, rdist, CUFFT_C2R, batch, &work_size),
                "plan_cufft<float>::cufftMakePlanMany() (backward)"
            );
        }
    }
    //! \brief Destructor, deletes the plan.
    ~plan_cufft(){ cufftDestroy(plan); }
    //! \brief Custom conversion to the cufftHandle.
    operator cufftHandle() const{ return plan; }

private:
    //! \brief The cufft opaque structure (pointer to struct).
    cufftHandle plan;
};

/*!
 * \ingroup hefftecuda
 * \brief Plan for the r2c single precision transform.
 */
template<> struct plan_cufft<double>{
    //! \brief Identical to the float specialization.
    plan_cufft(direction dir, int size, int batch, int stride, int rdist, int cdist){
        size_t work_size = 0;
        cuda::check_error(cufftCreate(&plan), "plan_cufft<float>::cufftCreate()");

        if (dir == direction::forward){
            cuda::check_error(
                cufftMakePlanMany(plan, 1, &size, &size, stride, rdist, &size, stride, cdist, CUFFT_D2Z, batch, &work_size),
                "plan_cufft<float>::cufftMakePlanMany() (forward)"
            );
        }else{
            cuda::check_error(
                cufftMakePlanMany(plan, 1, &size, &size, stride, cdist, &size, stride, rdist, CUFFT_Z2D, batch, &work_size),
                "plan_cufft<float>::cufftMakePlanMany() (backward)"
            );
        }
    }
    //! \brief Destructor, deletes the plan.
    ~plan_cufft(){ cufftDestroy(plan); }
    //! \brief Custom conversion to the cufftHandle.
    operator cufftHandle() const{ return plan; }

private:
    //! \brief The cufft opaque structure (pointer to struct).
    cufftHandle plan;
};

/*!
 * \ingroup hefftecuda
 * \brief Wrapper to cuFFT API for real-to-complex transform with shortening of the data.
 *
 * Serves the same purpose of heffte::cufft_executor but only real input is accepted
 * and only the unique (non-conjugate) coefficients are computed.
 * All real arrays must have size of real_size() and all complex arrays must have size complex_size().
 */
class cufft_executor_r2c{
public:
    /*!
     * \brief Constructor defines the box and the dimension of reduction.
     *
     * Note that the result sits in the box returned by box.r2c(dimension).
     */
    template<typename index>
    cufft_executor_r2c(box3d<index> const box, int dimension) :
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
    void forward(float const indata[], std::complex<float> outdata[]) const{
        make_plan(sforward, direction::forward);
        if (blocks == 1 or rblock_stride % 2 == 0){
            for(int i=0; i<blocks; i++){
                cufftReal *rdata = const_cast<cufftReal*>(indata + i * rblock_stride);
                cufftComplex* cdata = reinterpret_cast<cufftComplex*>(outdata + i * cblock_stride);
                cuda::check_error(cufftExecR2C(*sforward, rdata, cdata), "cufft_executor::cufftExecR2C()");
            }
        }else{
            // need to create a temporary copy of the data since cufftExecR2C() requires aligned input
            gpu::vector<float> rdata(rblock_stride);
            for(int i=0; i<blocks; i++){
                gpu::transfer::copy(indata + i * rblock_stride, rdata);
                cufftComplex* cdata = reinterpret_cast<cufftComplex*>(outdata + i * cblock_stride);
                cuda::check_error(cufftExecR2C(*sforward, rdata.data(), cdata), "cufft_executor::cufftExecR2C()");
            }
        }
    }
    //! \brief Backward transform, single precision.
    void backward(std::complex<float> const indata[], float outdata[]) const{
        make_plan(sbackward, direction::backward);
        if (blocks == 1 or rblock_stride % 2 == 0){
            for(int i=0; i<blocks; i++){
                cufftComplex* cdata = const_cast<cufftComplex*>(reinterpret_cast<cufftComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecC2R(*sbackward, cdata, outdata + i * rblock_stride), "cufft_executor::cufftExecC2R()");
            }
        }else{
            gpu::vector<float> odata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cufftComplex* cdata = const_cast<cufftComplex*>(reinterpret_cast<cufftComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecC2R(*sbackward, cdata, odata.data()), "cufft_executor::cufftExecC2R()");
                gpu::transfer::copy(odata, outdata + i * rblock_stride);
            }
        }
    }
    //! \brief Forward transform, double precision.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        make_plan(dforward, direction::forward);
        if (blocks == 1 or rblock_stride % 2 == 0){
            for(int i=0; i<blocks; i++){
                cufftDoubleReal *rdata = const_cast<cufftDoubleReal*>(indata + i * rblock_stride);
                cufftDoubleComplex* cdata = reinterpret_cast<cufftDoubleComplex*>(outdata + i * cblock_stride);
                cuda::check_error(cufftExecD2Z(*dforward, rdata, cdata), "cufft_executor::cufftExecD2Z()");
            }
        }else{
            gpu::vector<double> rdata(rblock_stride);
            for(int i=0; i<blocks; i++){
                gpu::transfer::copy(indata + i * rblock_stride, rdata);
                cufftDoubleComplex* cdata = reinterpret_cast<cufftDoubleComplex*>(outdata + i * cblock_stride);
                cuda::check_error(cufftExecD2Z(*dforward, rdata.data(), cdata), "cufft_executor::cufftExecD2Z()");
            }
        }
    }
    //! \brief Backward transform, double precision.
    void backward(std::complex<double> const indata[], double outdata[]) const{
        make_plan(dbackward, direction::backward);
        if (blocks == 1 or rblock_stride % 2 == 0){
            for(int i=0; i<blocks; i++){
                cufftDoubleComplex* cdata = const_cast<cufftDoubleComplex*>(reinterpret_cast<cufftDoubleComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecZ2D(*dbackward, cdata, outdata + i * rblock_stride), "cufft_executor::cufftExecZ2D()");
            }
        }else{
            gpu::vector<double> odata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cufftDoubleComplex* cdata = const_cast<cufftDoubleComplex*>(reinterpret_cast<cufftDoubleComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecZ2D(*dbackward, cdata, odata.data()), "cufft_executor::cufftExecZ2D()");
                gpu::transfer::copy(odata, outdata + i * rblock_stride);
            }
        }
    }

    //! \brief Returns the size of the box with real data.
    int real_size() const{ return rsize; }
    //! \brief Returns the size of the box with complex coefficients.
    int complex_size() const{ return csize; }

private:
    //! \brief Helper template to initialize the plan.
    template<typename scalar_type>
    void make_plan(std::unique_ptr<plan_cufft<scalar_type>> &plan, direction dir) const{
        if (!plan) plan = std::unique_ptr<plan_cufft<scalar_type>>(new plan_cufft<scalar_type>(dir, size, howmanyffts, stride, rdist, cdist));
    }

    int size, howmanyffts, stride, blocks;
    int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_cufft<float>> sforward;
    mutable std::unique_ptr<plan_cufft<double>> dforward;
    mutable std::unique_ptr<plan_cufft<float>> sbackward;
    mutable std::unique_ptr<plan_cufft<double>> dbackward;
};

/*!
 * \ingroup hefftecuda
 * \brief Helper struct that defines the types and creates instances of one-dimensional executors.
 *
 * The struct is specialized for each backend.
 */
template<> struct one_dim_backend<backend::cufft>{
    //! \brief Defines the complex-to-complex executor.
    using type = cufft_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = cufft_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    template<typename index>
    static std::unique_ptr<cufft_executor> make(void*, box3d<index> const box, int dimension){
        return std::unique_ptr<cufft_executor>(new cufft_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    template<typename index>
    static std::unique_ptr<cufft_executor_r2c> make_r2c(void*, box3d<index> const box, int dimension){
        return std::unique_ptr<cufft_executor_r2c>(new cufft_executor_r2c(box, dimension));
    }
};

namespace cuda { // packer logic

/*!
 * \ingroup hefftecuda
 * \brief Performs a direct-pack operation for data sitting on the GPU device.
 *
 * Launches a CUDA kernel.
 */
template<typename scalar_type, typename index>
void direct_pack(index nfast, index nmid, index nslow, index line_stride, index plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup hefftecuda
 * \brief Performs a direct-unpack operation for data sitting on the GPU device.
 *
 * Launches a CUDA kernel.
 */
template<typename scalar_type, typename index>
void direct_unpack(index nfast, index nmid, index nslow, index line_stride, index plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup hefftecuda
 * \brief Performs a transpose-unpack operation for data sitting on the GPU device.
 *
 * Launches a CUDA kernel.
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
        cuda::direct_pack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, data, buffer);
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type, typename index>
    void unpack(pack_plan_3d<index> const &plan, scalar_type const buffer[], scalar_type data[]) const{
        cuda::direct_unpack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, buffer, data);
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
        cuda::transpose_unpack<scalar_type>(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride,
                                            plan.buff_line_stride, plan.buff_plane_stride, plan.map[0], plan.map[1], plan.map[2], buffer, data);
    }
};

/*!
 * \ingroup hefftecuda
 * \brief Specialization for the CPU case.
 */
template<> struct data_scaling<tag::gpu>{
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type, typename index>
    static void apply(index num_entries, scalar_type *data, double scale_factor){
        cuda::scale_data(static_cast<long long>(num_entries), data, scale_factor);
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
 * \ingroup hefftecuda
 * \brief Sets the default options for the cufft backend.
 */
template<> struct default_plan_options<backend::cufft>{
    //! \brief The reshape operations will not transpose the data.
    static const bool use_reorder = false;
};

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
