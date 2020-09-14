/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_VECTOR_H
#define HEFFTE_BACKEND_VECTOR_H

/*!
 * \ingroup fft3d
 * \addtogroup hefftegpu Common templates for the GPU backends
 *
 * Templates used by the different GPU backends, mostly related
 * to memory management.
 */

namespace heffte{

/*!
 * \ingroup hefftegpu
 * \brief GPU specific methods.
 */
namespace gpu {

    /*!
     * \ingroup hefftegpu
     * \brief Container that wraps around a raw device array.
     *
     * Wrapper around a device array that allows for data to be automatically
     * allocated and freed using RAII style of resource management.
     * The \b scalar_type defines the data-type and the \b memory_manager
     * is a class/struct that implements several static members, e.g.,
     * \code
     * struct cuda_memory_manager{
     *     void* allocate(size_t num_bytes);
     *     void free(void *pnrs);
     *     void host_to_device(void const *source, size_t num_bytes, void *destination);
     *     void device_to_device(void const *source, size_t num_bytes, void *destination);
     *     void device_to_host(void const *source, size_t num_bytes, void *destination);
     * };
     * \endcode
     * All methods accept number of bytes, so sized passed have to be multiplied by
     * sizeof(scalar_type). The free method can accept a nullptr. The device/host
     * methods transfer data between the host, device, or device to device.
     *
     * The device_vector class is movable and copiable (using deep-copy)
     * in both the constructor and assignment operator.
     */
    template<typename scalar_type, typename memory_manager>
    class device_vector{
    public:
        //! \brief Allocate a new vector with the given number of entries.
        device_vector(size_t num_entries = 0) :
            num(num_entries),
            device_data(reinterpret_cast<scalar_type*>(memory_manager::allocate(num_entries * sizeof(scalar_type))))
        {}
        //! \brief Copy a range of entries from the device into the vector.
        device_vector(scalar_type const *begin, scalar_type const *end) :
            device_vector(std::distance(begin, end)){
            memory_manager::device_to_device(begin, num * sizeof(scalar_type), device_data);
        }

        //! \brief Copy constructor, copy the data from other to this vector.
        device_vector(const device_vector<scalar_type, memory_manager>& other) :
            device_vector(other.num){
            memory_manager::device_to_device(other.device_data, num * sizeof(scalar_type), device_data);
        }
        //! \brief Move constructor, moves the data from \b other into this vector.
        device_vector(device_vector<scalar_type, memory_manager> &&other) :
            num(c11_exchange(other.num, 0)),
            device_data(c11_exchange(other.device_data, nullptr))
        {}

        //! \brief Captures ownership of the data in the raw-pointer, resets the pointer to null.
        device_vector(scalar_type* &&raw_pointer, size_t num_entries) :
            num(num_entries),
            device_data(c11_exchange(raw_pointer, nullptr))
        {}

        //! \brief Desructor, deletes all data.
        ~device_vector(){ memory_manager::free(device_data); }

        //! \brief Copy assignment, copies the data form \b other to this object.
        void operator =(device_vector<scalar_type, memory_manager> const &other){
            device_vector<scalar_type, memory_manager> temp(other);
            std::swap(num, temp.num);
            std::swap(device_data, temp.device_data);
        }

        //! \brief Move assignment, moves the data form \b other to this object.
        void operator =(device_vector<scalar_type, memory_manager>&& other){
            device_vector<scalar_type, memory_manager> temp(std::move(other));
            std::swap(num, temp.num);
            std::swap(device_data, temp.device_data);
        }

        //! \brief Give reference to the array, can be passed directly into cuFFT calls or custom kernels.
        scalar_type* data(){ return device_data; }
        //! \brief Give const reference to the array, can be passed directly into cuFFT calls or custom kernels.
        const scalar_type* data() const{ return device_data; }

        //! \brief Return the current size of the array, i.e., the number of elements.
        size_t size() const{ return num; }
        //! \brief Return \b true if the vector is has zero size.
        bool empty() const{ return (num == 0); }

        //! \brief The value of the array, used for static error checking.
        using value_type = scalar_type;

        //! \brief Returns the current array and releases ownership.
        scalar_type* release(){
            num = 0;
            return c11_exchange(device_data, nullptr);
        }

    private:
        //! \brief Stores the number of entries in the vector.
        size_t num;
        //! \brief The array with the GPU data.
        scalar_type *device_data;
    };

    /*!
     * \ingroup hefftegpu
     * \brief Collection of helper methods that transfer data using a memory manager.
     *
     * Helper templates associated with a specific memory_manager, see device_vector.
     */
    template<typename memory_manager>
    struct device_transfer{
        //! \brief Copy data from the vector to the pointer, data size is equal to the vector size.
        template<typename scalar_type>
        static void copy(device_vector<scalar_type, memory_manager> const &source, scalar_type destination[]){
            memory_manager::device_to_host(source.data(), source.size() * sizeof(scalar_type), destination);
        }
        //! \brief Copy data from the pointer to the vector, data size is equal to the vector size.
        template<typename scalar_type>
        static void copy(scalar_type const source[], device_vector<scalar_type, memory_manager> &destination){
            memory_manager::device_to_host(source, destination.size() * sizeof(scalar_type), destination.data());
        }

        /*!
         * \brief Copy the data from a buffer on the CPU to a cuda::vector.
         *
         * \tparam scalar_type of the vector entries.
         *
         * \param cpu_source is a buffer with size at least \b num_entries that sits in the CPU
         * \param num_entries is the number of entries to load
         *
         * \returns a device_vector with size equal to \b num_entries and a copy of the CPU data
         */
        template<typename scalar_type>
        static device_vector<scalar_type, memory_manager> load(scalar_type const *cpu_source, size_t num_entries){
            device_vector<scalar_type, memory_manager> result(num_entries);
            memory_manager::host_to_device(cpu_source, num_entries * sizeof(scalar_type), result.data());
            return result;
        }
        //! \brief Similar to gpu::load() but loads the data from a std::vector
        template<typename scalar_type>
        static device_vector<scalar_type, memory_manager> load(std::vector<scalar_type> const &cpu_source){
            return load<scalar_type>(cpu_source.data(), cpu_source.size());
        }
        //! \brief Similar to gpu::load() but loads the data from a std::vector into a pointer.
        template<typename scalar_type>
        static void load(std::vector<scalar_type> const &cpu_source, scalar_type gpu_destination[]){
            memory_manager::host_to_device(cpu_source.data(), cpu_source.size() * sizeof(scalar_type), gpu_destination);
        }
        //! \brief Similar to gpu::load() but loads the data from a std::vector
        template<typename scalar_type>
        static void load(std::vector<scalar_type> const &cpu_source, device_vector<scalar_type, memory_manager> &gpu_destination){
            gpu_destination = load<scalar_type>(cpu_source);
        }

        /*!
         * \brief Load method that copies two std::vectors, used in template general code.
         *
         * This is never executed.
         * Without if-constexpr (introduced in C++ 2017) generic template code must compile
         * even branches in the if-statements that will never be reached.
         */
        template<typename scalar_type>
        static void load(std::vector<scalar_type> const &a, std::vector<scalar_type> &b){ b = a; }
        /*!
         * \brief Unload method that copies two std::vectors, used in template general code.
         *
         * This is never executed.
         * Without if-constexpr (introduced in C++ 2017) generic template code must compile
         * even branches in the if-statements that will never be reached.
         */
        template<typename scalar_type>
        static std::vector<scalar_type> unload(std::vector<scalar_type> const &a){ return a; }

        //! \brief Copy number of entries from the GPU pointer into the vector.
        template<typename scalar_type>
        static std::vector<scalar_type> unload(scalar_type const gpu_source[], size_t num_entries){
            std::vector<scalar_type> result(num_entries);
            memory_manager::device_to_host(gpu_source, num_entries * sizeof(scalar_type), result.data());
            return result;
        }

        /*!
         * \brief Copy the data from a cuda::vector to a cpu buffer
         *
         * \tparam scalar_type of the vector entries
         *
         * \param gpu_source is the cuda::vector to holding the data to unload
         * \param cpu_result is a buffer with size at least \b gpu_data.size() that sits in the CPU
         */
        template<typename scalar_type>
        static void unload(device_vector<scalar_type, memory_manager> const &gpu_source, scalar_type *cpu_result){
            memory_manager::device_to_host(gpu_source.data(), gpu_source.size() * sizeof(scalar_type), cpu_result);
        }

        //! \brief Similar to unload() but copies the data into a std::vector.
        template<typename scalar_type>
        static std::vector<scalar_type> unload(device_vector<scalar_type, memory_manager> const &gpu_source){
            std::vector<scalar_type> result(gpu_source.size());
            unload(gpu_source, result.data());
            return result;
        }
        /*!
         * \brief Captures ownership of the data in the raw-pointer.
         *
         * The advantage of the factory function over using the constructor is the ability
         * to auto-deduce the scalar type.
         */
        template<typename scalar_type>
        static device_vector<scalar_type, memory_manager> capture(scalar_type* &&raw_pointer, size_t num_entries){
            return device_vector<scalar_type, memory_manager>(std::forward<scalar_type*>(raw_pointer), num_entries);
        }
    };

    /*!
     * \ingroup hefftegpu
     * \brief Wrapper around cudaGetDeviceCount()
     */
    int device_count();

    /*!
     * \ingroup hefftegpu
     * \brief Wrapper around cudaSetDevice()
     *
     * \param active_device is the new active CUDA device for this thread, see the Nvidia documentation for cudaSetDevice()
     */
    void device_set(int active_device);

    /*!
     * \ingroup hefftegpu
     * \brief Wrapper around cudaStreamSynchronize(nullptr).
     */
    void synchronize_default_stream();
}

}

#endif   /* HEFFTE_BACKEND_VECTOR_H */
