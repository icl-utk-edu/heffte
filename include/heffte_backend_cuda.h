/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_CUDA_H
#define HEFFTE_BACKEND_CUDA_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_CUDA

#include <cufft.h>

namespace heffte{

//! \brief Replace with the C++ 2014 std::exchange later.
template<class T, class U = T>
T c11_exchange(T& obj, U&& new_value)
{
    T old_value = std::move(obj);
    obj = std::forward<U>(new_value);
    return old_value;
}

/*!
 * \brief Cuda specific methods, vector-like container, error checking, etc.
 */
namespace cuda {

    /*!
     * \brief Checks the status of a cuda command and in case of a failure, converts it to a C++ exception.
     */
    void check_error(cudaError_t status, std::string const &function_name);
    /*!
     * \brief Checks the status of a cufft command and in case of a failure, converts it to a C++ exception.
     */
    inline void check_error(cufftResult status, std::string const &function_name){
        if (status != CUFFT_SUCCESS)
            throw std::runtime_error(function_name + " failed with error code: " + std::to_string(status));
    }

    /*!
     * \brief Container that wraps around a raw cuda array.
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
        //! \brief Allocate a new cuda array with the given size.
        static scalar_type* alloc(size_t new_size);

    private:
        //! \brief Stores the number of entries in the vector.
        size_t num;
        //! \brief The array with the GPU data.
        scalar_type *gpu_data;
    };

    /*!
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
     * \brief Copy data from the vector to the pointer, data size is equal to the vector size.
     */
    template<typename scalar_type>
    void copy_pntr(vector<scalar_type> const &x, scalar_type data[]);
    /*!
     * \brief Copy data from the pointer to the vector, data size is equal to the vector size.
     */
    template<typename scalar_type>
    void copy_pntr(scalar_type const data[], vector<scalar_type> &x);

    /*!
     * \brief Copy the data from a buffer on the CPU to a cuda::vector.
     *
     * \tparam scalar_type of the vector entries.
     *
     * \param cpu_data is a buffer with size at least \b num_entries that sits in the CPU
     * \param num_entries is the number of entries to load
     *
     * \returns a cuda::vector with size equal to \b num_entries and a copy of the CPU data
     */
    template<typename scalar_type>
    vector<scalar_type> load(scalar_type const *cpu_data, size_t num_entries);
    /*!
     * \brief Similar to cuda::load() but loads the data from a std::vector
     */
    template<typename scalar_type>
    vector<scalar_type> load(std::vector<scalar_type> const &cpu_data){
        return load(cpu_data.data(), cpu_data.size());
    }
    /*!
     * \brief Similar to cuda::load() but loads the data from a std::vector into a pointer.
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &cpu_data, scalar_type gpu_data[]);
    /*!
     * \brief Similar to cuda::load() but loads the data from a std::vector
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &cpu_data, vector<scalar_type> &gpu_data);

    /*!
     * \brief Load method that copies two std::vectors, used in template general code.
     *
     * This is never executed.
     * Without if-constexpr (introduced in C++ 2017) generic template code must compile
     * even branches in the if-statements that will never be reached.
     */
    template<typename scalar_type>
    void load(std::vector<scalar_type> const &a, std::vector<scalar_type> &b){ b = a; }
    /*!
     * \brief Unload method that copies two std::vectors, used in template general code.
     *
     * This is never executed.
     * Without if-constexpr (introduced in C++ 2017) generic template code must compile
     * even branches in the if-statements that will never be reached.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(std::vector<scalar_type> const &a){ return a; }

    /*!
     * \brief Copy number of entries from the GPU pointer into the vector.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(scalar_type const gpu_pointer[], size_t num_entries);

    /*!
     * \brief Copy the data from a cuda::vector to a cpu buffer
     *
     * \tparam scalar_type of the vector entries
     *
     * \param gpu_data is the cuda::vector to holding the data to unload
     * \param cpu_data is a buffer with size at least \b gpu_data.size() that sits in the CPU
     */
    template<typename scalar_type>
    void unload(vector<scalar_type> const &gpu_data, scalar_type *cpu_data);

    /*!
     * \brief Similar to unload() but copies the data into a std::vector.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(vector<scalar_type> const &gpu_data){
        std::vector<scalar_type> result(gpu_data.size());
        unload(gpu_data, result.data());
        return result;
    }

    /*!
     * \brief Convert real numbers to complex when both are located on the GPU device.
     *
     * Launches a CUDA kernel.
     */
    template<typename precision_type>
    void convert(int num_entries, precision_type const source[], std::complex<precision_type> destination[]);
    /*!
     * \brief Convert complex numbers to real when both are located on the GPU device.
     *
     * Launches a CUDA kernel.
     */
    template<typename precision_type>
    void convert(int num_entries, std::complex<precision_type> const source[], precision_type destination[]);

    /*!
     * \brief Scales real data (double or float) by the scaling factor.
     */
    template<typename scalar_type>
    void scale_data(int num_entries, scalar_type *data, double scale_factor);
}

/*!
 * \brief Data manipulations on the GPU end.
 */
template<> struct data_manipulator<tag::gpu>{
    /*!
     * \brief Equivalent to std::copy_n() but using CUDA arrays.
     */
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]);
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void scale(int num_entries, scalar_type data[], double scale_factor){
        cuda::scale_data(num_entries, data, scale_factor);
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
    //! \brief Type-tag for the cuFFT backend
    struct cufft{};

    //! \brief Indicate that the cuFFT backend has been enabled.
    template<> struct is_enabled<cufft> : std::true_type{};

    /*!
     * \brief Defines the location type-tag and the cuda container.
     */
    template<>
    struct buffer_traits<cufft>{
        //! \brief The cufft library uses data on the gpu device.
        using location = tag::gpu;
        //! \brief The data is managed by the cuda vector container.
        template<typename T> using container = heffte::cuda::vector<T>;
    };

    /*!
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<cufft>(){ return "cufft"; }
}

/*!
 * \brief Recognize the cuFFT single precision complex type.
 */
template<> struct is_ccomplex<cufftComplex> : std::true_type{};
/*!
 * \brief Recognize the cuFFT double precision complex type.
 */
template<> struct is_zcomplex<cufftDoubleComplex> : std::true_type{};

/*!
 * \brief Base plan for cufft, using only the specialization for float and double complex.
 *
 * Similar to heffte::plan_fftw but applies to the cufft backend.
 */
template<typename> struct plan_cufft{};

/*!
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
    mutable cufftHandle plan;
};
//! \brief Specialization for double complex.
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
    mutable cufftHandle plan;
};

/*!
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
    cufft_executor(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == 0) ? size : 1),
        blocks((dimension == 1) ? box.size[2] : 1),
        block_stride(box.size[0] * box.size[1]),
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

    mutable int size, howmanyffts, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_cufft<std::complex<float>>> ccomplex_plan;
    mutable std::unique_ptr<plan_cufft<std::complex<double>>> zcomplex_plan;
};

/*!
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
    mutable cufftHandle plan;
};

/*!
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
    mutable cufftHandle plan;
};

/*!
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
    cufft_executor_r2c(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        blocks((dimension == 1) ? box.size[2] : 1),
        rdist((dimension == 0) ? size : 1),
        cdist((dimension == 0) ? size/2 + 1 : 1),
        rblock_stride(box.size[0] * box.size[1]),
        cblock_stride(box.size[0] * (box.size[1]/2 + 1)),
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
            cuda::vector<float> rdata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cuda::copy_pntr(indata + i * rblock_stride, rdata);
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
            cuda::vector<float> odata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cufftComplex* cdata = const_cast<cufftComplex*>(reinterpret_cast<cufftComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecC2R(*sbackward, cdata, odata.data()), "cufft_executor::cufftExecC2R()");
                cuda::copy_pntr(odata, outdata + i * rblock_stride);
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
            cuda::vector<double> rdata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cuda::copy_pntr(indata + i * rblock_stride, rdata);
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
            cuda::vector<double> odata(rblock_stride);
            for(int i=0; i<blocks; i++){
                cufftDoubleComplex* cdata = const_cast<cufftDoubleComplex*>(reinterpret_cast<cufftDoubleComplex const*>(indata + i * cblock_stride));
                cuda::check_error(cufftExecZ2D(*dbackward, cdata, odata.data()), "cufft_executor::cufftExecZ2D()");
                cuda::copy_pntr(odata, outdata + i * rblock_stride);
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

    mutable int size, howmanyffts, stride, blocks;
    mutable int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_cufft<float>> sforward;
    mutable std::unique_ptr<plan_cufft<double>> dforward;
    mutable std::unique_ptr<plan_cufft<float>> sbackward;
    mutable std::unique_ptr<plan_cufft<double>> dbackward;
};

/*!
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
    static std::unique_ptr<cufft_executor> make(box3d const box, int dimension){
        return std::unique_ptr<cufft_executor>(new cufft_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    static std::unique_ptr<cufft_executor_r2c> make_r2c(box3d const box, int dimension){
        return std::unique_ptr<cufft_executor_r2c>(new cufft_executor_r2c(box, dimension));
    }
};

namespace cuda { // packer logic

/*!
 * \brief Performs a direct-pack operation for data sitting on the GPU device.
 *
 * Launches a CUDA kernel.
 */
template<typename scalar_type>
void direct_pack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]);
/*!
 * \brief Performs a direct-unpack operation for data sitting on the GPU device.
 *
 * Launches a CUDA kernel.
 */
template<typename scalar_type>
void direct_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]);

}

/*!
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::gpu>{
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        cuda::direct_pack(plan.nfast, plan.nmid, plan.nslow, plan.line_stride, plan.plane_stride, data, buffer);
    }
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        cuda::direct_unpack(plan.nfast, plan.nmid, plan.nslow, plan.line_stride, plan.plane_stride, buffer, data);
    }
};

/*!
 * \brief Specialization for the CPU case.
 */
template<> struct data_scaling<tag::gpu>{
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void apply(int num_entries, scalar_type *data, double scale_factor){
        cuda::scale_data(num_entries, data, scale_factor);
    }
    /*!
     * \brief Complex by real scaling.
     */
    template<typename precision_type>
    static void apply(int num_entries, std::complex<precision_type> *data, double scale_factor){
        apply<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
