/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_ROCM_H
#define HEFFTE_BACKEND_ROCM_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_ROCM

#include <rocfft.h>

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
 * \brief Rocm specific methods, vector-like container, error checking, etc.
 */
namespace rocm {

    /*!
     * \ingroup heffterocm
     * \brief Checks the status of a rocm command and in case of a failure, converts it to a C++ exception.
     */
    //void check_error(hipError_t status, std::string const &function_name);
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
     * \brief Wrapper around hipGetDeviceCount()
     */
    int device_count();

    /*!
     * \ingroup heffterocm
     * \brief Wrapper around hipSetDevice()
     *
     * \param active_device is the new active ROCM device for this thread, see the AMD documentation for hipSetDevice()
     */
    void device_set(int active_device);

    /*!
     * \ingroup heffterocm
     * \brief Wrapper around hipStreamSynchronize(nullptr).
     */
    void synchronize_default_stream();

    /*!
     * \ingroup heffterocm
     * \brief Container that wraps around a raw ROCM array.
     */
    template<typename scalar_type> class vector{
    public:
        //! \brief Allocate a new vector with the given number of entries.
        vector(size_t num_entries = 0) : num(num_entries), gpu_data(alloc(num)){}
        //! \brief Copy a range of entries from the device into the vector.
        vector(scalar_type const *begin, scalar_type const *end);

        //! \brief Copy constructor, copy the data from other to this vector.
        vector(const vector<scalar_type>& other);
        //! \brief Move constructor, moves the data from \b other into this vector.
        vector(vector<scalar_type> &&other) : num(c11_exchange(other.num, 0)), gpu_data(c11_exchange(other.gpu_data, nullptr)){}

        //! \brief Captures ownership of the data in the raw-pointer, resets the pointer to null.
        vector(scalar_type* &&raw_pointer, size_t num_entries) : num(num_entries), gpu_data(c11_exchange(raw_pointer, nullptr)){}

        //! \brief Desructor, deletes all data.
        ~vector();

        //! \brief Copy assignment, copies the data form \b other to this object.
        void operator =(vector<scalar_type> const &other){
            vector<scalar_type> temp(other);
            std::swap(num, temp.num);
            std::swap(gpu_data, temp.gpu_data);
        }

        //! \brief Move assignment, moves the data form \b other to this object.
        void operator =(vector<scalar_type>&& other){
            vector<scalar_type> temp(std::move(other));
            std::swap(num, temp.num);
            std::swap(gpu_data, temp.gpu_data);
        }

        //! \brief Give reference to the array, can be passed directly into cuFFT calls or custom kernels.
        scalar_type* data(){ return gpu_data; }
        //! \brief Give const reference to the array, can be passed directly into cuFFT calls or custom kernels.
        const scalar_type* data() const{ return gpu_data; }

        //! \brief Return the current size of the array, i.e., the number of elements.
        size_t size() const{ return num; }
        //! \brief Return \b true if the vector is has zero size.
        bool empty() const{ return (num == 0); }

        //! \brief The value of the array, used for static error checking.
        using value_type = scalar_type;

        //! \brief Returns the current array and releases ownership.
        scalar_type* release(){
            num = 0;
            return c11_exchange(gpu_data, nullptr);
        }

    protected:
        //! \brief Allocate a new ROCM array with the given size.
        static scalar_type* alloc(size_t new_size);

    private:
        //! \brief Stores the number of entries in the vector.
        size_t num;
        //! \brief The array with the GPU data.
        scalar_type *gpu_data;
    };

    /*!
     * \ingroup heffterocm
     * \brief Captures ownership of the data in the raw-pointer.
     *
     * The advantage of the factory function over using the constructor is the ability
     * to auto-deduce the scalar type.
     */
    template<typename scalar_type>
    vector<scalar_type> capture(scalar_type* &&raw_pointer, size_t num_entries){
        return vector<scalar_type>(std::forward<scalar_type*>(raw_pointer), num_entries);
    }

    /*!
     * \ingroup heffterocm
     * \brief Copy data from the vector to the pointer, data size is equal to the vector size.
     */
    template<typename scalar_type>
    void copy_pntr(vector<scalar_type> const &x, scalar_type data[]);
    /*!
     * \ingroup heffterocm
     * \brief Copy data from the pointer to the vector, data size is equal to the vector size.
     */
    template<typename scalar_type>
    void copy_pntr(scalar_type const data[], vector<scalar_type> &x);

    /*!
     * \ingroup heffterocm
     * \brief Copy the data from a buffer on the CPU to a rocm::vector.
     *
     * \tparam scalar_type of the vector entries.
     *
     * \param cpu_data is a buffer with size at least \b num_entries that sits in the CPU
     * \param num_entries is the number of entries to load
     *
     * \returns a rocm::vector with size equal to \b num_entries and a copy of the CPU data
     */
    template<typename scalar_type>
    vector<scalar_type> load(scalar_type const *cpu_data, size_t num_entries);
    /*!
     * \ingroup heffterocm
     * \brief Similar to rocm::load() but loads the data from a std::vector
     */
    template<typename scalar_type>
    vector<scalar_type> load(std::vector<scalar_type> const &cpu_data){
        return load(cpu_data.data(), cpu_data.size());
    }
    /*!
     * \ingroup heffterocm
     * \brief Similar to rocm::load() but loads the data from a std::vector into a pointer.
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &cpu_data, scalar_type gpu_data[]);
    /*!
     * \ingroup heffterocm
     * \brief Similar to rocm::load() but loads the data from a std::vector
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &cpu_data, vector<scalar_type> &gpu_data);

    /*!
     * \ingroup heffterocm
     * \brief Load method that copies two std::vectors, used in template general code.
     *
     * This is never executed.
     * Without if-constexpr (introduced in C++ 2017) generic template code must compile
     * even branches in the if-statements that will never be reached.
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &a, std::vector<scalar_type> &b){ b = a; }
    /*!
     * \ingroup heffterocm
     * \brief Unload method that copies two std::vectors, used in template general code.
     *
     * This is never executed.
     * Without if-constexpr (introduced in C++ 2017) generic template code must compile
     * even branches in the if-statements that will never be reached.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(std::vector<scalar_type> const &a){ return a; }

    /*!
     * \ingroup heffterocm
     * \brief Copy number of entries from the GPU pointer into the vector.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(scalar_type const gpu_pointer[], size_t num_entries);

    /*!
     * \ingroup heffterocm
     * \brief Copy the data from a rocm::vector to a cpu buffer
     *
     * \tparam scalar_type of the vector entries
     *
     * \param gpu_data is the rocm::vector to holding the data to unload
     * \param cpu_data is a buffer with size at least \b gpu_data.size() that sits in the CPU
     */
    template<typename scalar_type>
    void unload(vector<scalar_type> const &gpu_data, scalar_type *cpu_data);

    /*!
     * \ingroup heffterocm
     * \brief Similar to unload() but copies the data into a std::vector.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(vector<scalar_type> const &gpu_data){
        std::vector<scalar_type> result(gpu_data.size());
        unload(gpu_data, result.data());
        return result;
    }

    /*!
     * \ingroup heffterocm
     * \brief Convert real numbers to complex when both are located on the GPU device.
     *
     * Launches a ROCM kernel.
     */
    template<typename precision_type>
    void convert(int num_entries, precision_type const source[], std::complex<precision_type> destination[]);
    /*!
     * \ingroup heffterocm
     * \brief Convert complex numbers to real when both are located on the GPU device.
     *
     * Launches a ROCM kernel.
     */
    template<typename precision_type>
    void convert(int num_entries, std::complex<precision_type> const source[], precision_type destination[]);

    /*!
     * \ingroup heffterocm
     * \brief Scales real data (double or float) by the scaling factor.
     */
    template<typename scalar_type>
    void scale_data(int num_entries, scalar_type *data, double scale_factor);
}

/*!
 * \ingroup heffterocm
 * \brief Data manipulations on the GPU end.
 */
template<> struct data_manipulator<tag::gpu>{
    /*!
     * \brief Equivalent to std::copy_n() but using ROCM arrays.
     */
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]);
    //! \brief Copy-convert complex-to-real.
    template<typename scalar_type>
    static void copy_n(std::complex<scalar_type> const source[], size_t num_entries, scalar_type destination[]){
        rocm::convert(num_entries, source, destination);
    }
    //! \brief Copy-convert real-to-complex.
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, std::complex<scalar_type> destination[]){
        rocm::convert(num_entries, source, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void scale(int num_entries, scalar_type data[], double scale_factor){
        rocm::scale_data(num_entries, data, scale_factor);
    }
    /*!
     * \brief Complex by real scaling.
     */
    template<typename precision_type>
    static void scale(int num_entries, std::complex<precision_type> data[], double scale_factor){
        scale<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

namespace backend{
    /*!
     * \ingroup heffterocm
     * \brief Type-tag for the rocFFT backend
     */
    struct rocfft{};

    /*!
     * \ingroup heffterocm
     * \brief Indicate that the cuFFT backend has been enabled.
     */
    template<> struct is_enabled<rocfft> : std::true_type{};

    /*!
     * \ingroup heffterocm
     * \brief Defines the location type-tag and the ROCM container.
     */
    template<>
    struct buffer_traits<rocfft>{
        //! \brief The rocfft library uses data on the gpu device.
        using location = tag::gpu;
        //! \brief The data is managed by the ROCM vector container.
        template<typename T> using container = heffte::rocm::vector<T>;
    };

    /*!
     * \ingroup heffterocm
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<rocfft>(){ return "rocfft"; }
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
     * \brief Constructor, takes inputs identical to cufftMakePlanMany().
     *
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
    rocfft_executor(box3d const box, int dimension) :
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
        //rocm::synchronize_default_stream();
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

        size_t wsize = (std::is_same<precision_type, float>::value) ?
                            ((dir == direction::forward) ? ccomplex_forward->size_work() : ccomplex_backward->size_work()) :
                            ((dir == direction::forward) ? zcomplex_forward->size_work() : zcomplex_backward->size_work());
        rocm::vector<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = rocm::vector<std::complex<precision_type>>(wsize);
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
        //rocm::synchronize_default_stream();
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
        rocm::convert(total_size, indata, outdata);
        forward(outdata);
    }
    //! \brief Performs backward float-complex transform and truncates the complex part of the result.
    template<typename precision_type>
    void backward(std::complex<precision_type> indata[], precision_type outdata[]) const{
        backward(indata);
        rocm::convert(total_size, indata, outdata);
    }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    //! \brief Helper template to create the plan.
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_rocfft<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_rocfft<scalar_type, dir>>(new plan_rocfft<scalar_type, dir>(size, howmanyffts, stride, dist));
    }

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
    rocfft_executor_r2c(box3d const box, int dimension) :
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

        size_t wsize = (std::is_same<precision_type, float>::value) ? sforward->size_work() : dforward->size_work();
        rocm::vector<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = rocm::vector<std::complex<precision_type>>(wsize);
            rocfft_execution_info_set_work_buffer(info, reinterpret_cast<void*>(work_buff.data()), wsize);
        }
        rocm::vector<precision_type> copy_indata(indata, indata + real_size());

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
        rocm::vector<std::complex<precision_type>> work_buff;

        if (wsize > 0){
            work_buff = rocm::vector<std::complex<precision_type>>(wsize);
            rocfft_execution_info_set_work_buffer(info, reinterpret_cast<void*>(work_buff.data()), wsize);
        }
        rocm::vector<std::complex<precision_type>> copy_indata(indata, indata + complex_size());

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
    static std::unique_ptr<rocfft_executor> make(box3d const box, int dimension){
        return std::unique_ptr<rocfft_executor>(new rocfft_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    static std::unique_ptr<rocfft_executor_r2c> make_r2c(box3d const box, int dimension){
        return std::unique_ptr<rocfft_executor_r2c>(new rocfft_executor_r2c(box, dimension));
    }
};

namespace rocm { // packer logic

/*!
 * \ingroup heffterocm
 * \brief Performs a direct-pack operation for data sitting on the GPU device.
 *
 * Launches a HIP kernel.
 */
template<typename scalar_type>
void direct_pack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup heffterocm
 * \brief Performs a direct-unpack operation for data sitting on the GPU device.
 *
 * Launches a HIP kernel.
 */
template<typename scalar_type>
void direct_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \ingroup heffterocm
 * \brief Performs a transpose-unpack operation for data sitting on the GPU device.
 *
 * Launches a HIP kernel.
 */
template<typename scalar_type>
void transpose_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stide,
                      int buff_line_stride, int buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]);

}

/*!
 * \ingroup hefftepacking
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        rocm::direct_pack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, data, buffer);
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        rocm::direct_unpack(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride, buffer, data);
    }
};

/*!
 * \ingroup hefftepacking
 * \brief GPU version of the transpose packer.
 */
template<> struct transpose_packer<tag::gpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        direct_packer<tag::gpu>().pack(plan, data, buffer); // packing is done the same way as the direct_packer
    }
    //! \brief Execute the planned transpose-unpack operation.
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        rocm::transpose_unpack<scalar_type>(plan.size[0], plan.size[1], plan.size[2], plan.line_stride, plan.plane_stride,
                                            plan.buff_line_stride, plan.buff_plane_stride, plan.map[0], plan.map[1], plan.map[2], buffer, data);
    }
};

/*!
 * \ingroup heffterocm
 * \brief Specialization for the CPU case.
 */
template<> struct data_scaling<tag::gpu>{
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void apply(int num_entries, scalar_type *data, double scale_factor){
        rocm::scale_data(num_entries, data, scale_factor);
    }
    /*!
     * \brief Complex by real scaling.
     */
    template<typename precision_type>
    static void apply(int num_entries, std::complex<precision_type> *data, double scale_factor){
        apply<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
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
