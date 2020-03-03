/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_CUDA_H
#define HEFFTE_BACKEND_CUDA_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_CUDA

#include <cuda_runtime_api.h>
#include <cuda.h>
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
    inline void check_error(cudaError_t status, std::string const &function_name){
        if (status != cudaSuccess)
            throw std::runtime_error(function_name + " failed with message: " + cudaGetErrorString(status));
    }

    /*!
     * \brief Container that wraps around a raw cuda array.
     */
    template<typename scalar_type> class vector{
    public:
        //! \brief Allocate a new vector with the given number of entries.
        vector(size_t num_entries = 0) : num(num_entries), gpu_data(alloc(num)){}

        //! \brief Copy a range of entries from the device into the vector.
        vector(scalar_type const *begin, scalar_type const *end) : num(std::distance(begin, end)), gpu_data(alloc(num)){
            check_error(cudaMemcpy(gpu_data, begin, num * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::vector(begin, end)");
        }

        //! \brief Copy constructor, copy the data from other to this vector.
        vector(const vector<scalar_type>& other) : num(other.num), gpu_data(alloc(num)){
            check_error(cudaMemcpy(gpu_data, other.gpu_data, num * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::vector(cuda::vector const &)");
        }
        //! \brief Move constructor, moves the data from \b other into this vector.
        vector(vector<scalar_type> &&other) : num(c11_exchange(other.num, 0)), gpu_data(c11_exchange(other.gpu_data, nullptr)){}

        //! \brief Desructor, deletes all data.
        ~vector(){
            if (gpu_data != nullptr)
                check_error(cudaFree(gpu_data), "cuda::~vector()");
        }

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
        static scalar_type* alloc(size_t new_size){
            if (new_size == 0) return nullptr;
            scalar_type *new_data;
            check_error(cudaMalloc((void**) &new_data, new_size * sizeof(scalar_type)), "cuda::vector::alloc()");
            return new_data;
        }

    private:
        //! \brief Stores the number of entries in the vector.
        size_t num;
        //! \brief The array with the GPU data.
        scalar_type *gpu_data;
    };

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
    vector<scalar_type> load(scalar_type const *cpu_data, size_t num_entries){
        vector<scalar_type> result(num_entries);
        check_error(cudaMemcpy(result.data(), cpu_data, num_entries * sizeof(scalar_type), cudaMemcpyHostToDevice), "cuda::load()");
        return result;
    }
    /*!
     * \brief Similar to cuda::load() but loads the data from a std::vector
     */
    template<typename scalar_type>
    vector<scalar_type> load(std::vector<scalar_type> const &cpu_data){
        return load(cpu_data.data(), cpu_data.size());
    }

    /*!
     * \brief Copy the data from a cuda::vector to a cpu buffer
     *
     * \tparam scalar_type of the vector entries
     *
     * \param gpu_data is the cuda::vector to holding the data to unload
     * \param cpu_data is a buffer with size at least \b gpu_data.size() that sits in the CPU
     */
    template<typename scalar_type>
    void unload(vector<scalar_type> const &gpu_data, scalar_type *cpu_data){
        check_error(cudaMemcpy(cpu_data, gpu_data.data(), gpu_data.size() * sizeof(scalar_type), cudaMemcpyDeviceToHost), "cuda::unload()");
    }
    /*!
     * \brief Similar to unload() but copies the data into a std::vector.
     */
    template<typename scalar_type>
    std::vector<scalar_type> unload(vector<scalar_type> const &gpu_data){
        std::vector<scalar_type> result(gpu_data.size());
        unload(gpu_data, result.data());
        return result;
    }
}

namespace backend{
    //! \brief Type-tag for the cuFFT backend
    struct cufft{};

    //! \brief Indicate that the cuFFT backend has been enabled.
    template<> struct is_enabled<cufft> : std::true_type{};

    template<>
    struct buffer_traits<cufft>{
        using location = tag::gpu;
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

// /*!
//  * \brief Base plan for fftw, using only the specialization for float and double complex.
//  *
//  * FFTW3 library uses plans for forward and backward fft transforms.
//  * The specializations to this struct will wrap around such plans and provide RAII style
//  * of memory management and simple constructors that take inputs suitable to HeFFTe.
//  */
// template<typename, direction> struct plan_fftw{};
//
// /*!
//  * \brief Plan for the single precision complex transform.
//  *
//  * \tparam dir indicates a forward or backward transform
//  */
// template<direction dir>
// struct plan_fftw<std::complex<float>, dir>{
//     /*!
//      * \brief Constructor, takes inputs identical to fftwf_plan_many_dft().
//      *
//      * \param size is the number of entries in a 1-D transform
//      * \param howmany is the number of transforms in the batch
//      * \param stride is the distance between entries of the same transform
//      * \param dist is the distance between the first entries of consecutive sequences
//      */
//     plan_fftw(int size, int howmany, int stride, int dist) :
//         plan(fftwf_plan_many_dft(1, &size, howmany, nullptr, nullptr, stride, dist,
//                                                     nullptr, nullptr, stride, dist,
//                                                     (dir == direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE
//                                 ))
//         {}
//     //! \brief Destructor, deletes the plan.
//     ~plan_fftw(){ fftwf_destroy_plan(plan); }
//     //! \brief Custom conversion to the FFTW3 plan.
//     operator fftwf_plan() const{ return plan; }
//     //! \brief The FFTW3 opaque structure (pointer to struct).
//     fftwf_plan plan;
// };
// //! \brief Specialization for double complex.
// template<direction dir>
// struct plan_fftw<std::complex<double>, dir>{
//     //! \brief Identical to the float-complex specialization.
//     plan_fftw(int size, int howmany, int stride, int dist) :
//         plan(fftw_plan_many_dft(1, &size, howmany, nullptr, nullptr, stride, dist,
//                                                    nullptr, nullptr, stride, dist,
//                                                    (dir == direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE
//                                ))
//         {}
//     //! \brief Identical to the float-complex specialization.
//     ~plan_fftw(){ fftw_destroy_plan(plan); }
//     //! \brief Identical to the float-complex specialization.
//     operator fftw_plan() const{ return plan; }
//     //! \brief Identical to the float-complex specialization.
//     fftw_plan plan;
// };
//
//
// class fftw_executor{
// public:
//     fftw_executor(box3d const box, int dimension) :
//         size(box.size[dimension]),
//         howmany(get_many(box, dimension)),
//         stride(get_stride(box, dimension)),
//         dist((dimension == 0) ? size : 1),
//         blocks((dimension == 1) ? box.size[2] : 1),
//         block_stride(box.size[0] * box.size[1]),
//         total_size(box.count())
//     {}
//
//     static int get_many(box3d const box, int dimension){
//         if (dimension == 0) return box.size[1] * box.size[2];
//         if (dimension == 1) return box.size[0];
//         return box.size[0] * box.size[1];
//     }
//     static int get_stride(box3d const box, int dimension){
//         if (dimension == 0) return 1;
//         if (dimension == 1) return box.size[0];
//         return box.size[0] * box.size[1];
//     }
//
//     void forward(std::complex<float> data[]) const{
//         make_plan(cforward);
//         for(int i=0; i<blocks; i++){
//             fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
//             fftwf_execute_dft(*cforward, block_data, block_data);
//         }
//     }
//     void backward(std::complex<float> data[]) const{
//         make_plan(cbackward);
//         for(int i=0; i<blocks; i++){
//             fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
//             fftwf_execute_dft(*cbackward, block_data, block_data);
//         }
//     }
//     void forward(std::complex<double> data[]) const{
//         make_plan(zforward);
//         for(int i=0; i<blocks; i++){
//             fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
//             fftw_execute_dft(*zforward, block_data, block_data);
//         }
//     }
//     void backward(std::complex<double> data[]) const{
//         make_plan(zbackward);
//         for(int i=0; i<blocks; i++){
//             fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
//             fftw_execute_dft(*zbackward, block_data, block_data);
//         }
//     }
//
//     void forward(float const indata[], std::complex<float> outdata[]) const{
//         for(int i=0; i<total_size; i++) outdata[i] = std::complex<float>(indata[i]);
//         forward(outdata);
//     }
//     void backward(std::complex<float> indata[], float outdata[]) const{
//         backward(indata);
//         for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
//     }
//     void forward(double const indata[], std::complex<double> outdata[]) const{
//         for(int i=0; i<total_size; i++) outdata[i] = std::complex<double>(indata[i]);
//         forward(outdata);
//     }
//     void backward(std::complex<double> indata[], double outdata[]) const{
//         backward(indata);
//         for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
//     }
//
//     int box_size() const{ return total_size; }
//
// private:
//     template<typename scalar_type, direction dir>
//     void make_plan(std::unique_ptr<plan_fftw<scalar_type, dir>> &plan) const{
//         if (!plan) plan = std::unique_ptr<plan_fftw<scalar_type, dir>>(new plan_fftw<scalar_type, dir>(size, howmany, stride, dist));
//     }
//
//     mutable int size, howmany, stride, dist, blocks, block_stride, total_size;
//     mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::forward>> cforward;
//     mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::backward>> cbackward;
//     mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::forward>> zforward;
//     mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::backward>> zbackward;
// };
//
// template<> struct one_dim_backend<backend::cufft>{
//     using type = fftw_executor;
//
//     static std::unique_ptr<fftw_executor> make(box3d const box, int dimension){
//         return std::unique_ptr<fftw_executor>(new fftw_executor(box, dimension));
//     }
// };

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
