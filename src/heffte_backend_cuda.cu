/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_backend_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

namespace heffte {
namespace cuda {

void check_error(cudaError_t status, std::string const &function_name){
    if (status != cudaSuccess)
        throw std::runtime_error(function_name + " failed with message: " + cudaGetErrorString(status));
}

template<typename scalar_type>
vector<scalar_type>::vector(scalar_type const *begin, scalar_type const *end) : num(std::distance(begin, end)), gpu_data(alloc(num)){
    check_error(cudaMemcpy(gpu_data, begin, num * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::vector(begin, end)");
}
template<typename scalar_type>
vector<scalar_type>::vector(const vector<scalar_type>& other) : num(other.num), gpu_data(alloc(num)){
    check_error(cudaMemcpy(gpu_data, other.gpu_data, num * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::vector(cuda::vector const &)");
}
template<typename scalar_type>
vector<scalar_type>::~vector(){
    if (gpu_data != nullptr)
        check_error(cudaFree(gpu_data), "cuda::~vector()");
}
template<typename scalar_type>
scalar_type* vector<scalar_type>::alloc(size_t new_size){
    if (new_size == 0) return nullptr;
    scalar_type *new_data;
    check_error(cudaMalloc((void**) &new_data, new_size * sizeof(scalar_type)), "cuda::vector::alloc()");
    return new_data;
}
template<typename scalar_type>
void copy_pntr(vector<scalar_type> const &x, scalar_type data[]){
    check_error(cudaMemcpy(data, x.data(), x.size() * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::copy_pntr(vector, data)");
}
template<typename scalar_type>
void copy_pntr(scalar_type const data[], vector<scalar_type> &x){
    check_error(cudaMemcpy(x.data(), data, x.size() * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "cuda::copy_pntr(data, vector)");
}
template<typename scalar_type>
vector<scalar_type> load(scalar_type const *cpu_data, size_t num_entries){
    vector<scalar_type> result(num_entries);
    check_error(cudaMemcpy(result.data(), cpu_data, num_entries * sizeof(scalar_type), cudaMemcpyHostToDevice), "cuda::load()");
    return result;
}
template<typename scalar_type>
void load(std::vector<scalar_type> const &cpu_data, vector<scalar_type> &gpu_data){
    if (gpu_data.size() != cpu_data.size()) gpu_data = vector<scalar_type>(cpu_data.size());
    check_error(cudaMemcpy(gpu_data.data(), cpu_data.data(), gpu_data.size() * sizeof(scalar_type), cudaMemcpyHostToDevice), "cuda::load()");
}
template<typename scalar_type>
void unload(vector<scalar_type> const &gpu_data, scalar_type *cpu_data){
    check_error(cudaMemcpy(cpu_data, gpu_data.data(), gpu_data.size() * sizeof(scalar_type), cudaMemcpyDeviceToHost), "cuda::unload()");
}

#define instantiate_cuda_vector(scalar_type) \
    template vector<scalar_type>::vector(scalar_type const *begin, scalar_type const *end); \
    template vector<scalar_type>::vector(const vector<scalar_type>& other); \
    template vector<scalar_type>::~vector(); \
    template scalar_type* vector<scalar_type>::alloc(size_t); \
    template void copy_pntr(vector<scalar_type> const &x, scalar_type data[]); \
    template void copy_pntr(scalar_type const data[], vector<scalar_type> &x); \
    template vector<scalar_type> load(scalar_type const *cpu_data, size_t num_entries); \
    template void load<scalar_type>(std::vector<scalar_type> const &cpu_data, vector<scalar_type> &gpu_data); \
    template void unload<scalar_type>(vector<scalar_type> const &, scalar_type *); \


instantiate_cuda_vector(float);
instantiate_cuda_vector(double);
instantiate_cuda_vector(std::complex<float>);
instantiate_cuda_vector(std::complex<double>);

/*
 * Launch with one thread per entry.
 *
 * If to_complex is true, convert one real number from source to two real numbers in destination.
 * If to_complex is false, convert two real numbers from source to one real number in destination.
 */
template<typename scalar_type, int num_threads, bool to_complex>
__global__ void real_complex_convert(int num_entries, scalar_type const source[], scalar_type destination[]){
    int i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        if (to_complex){
            destination[2*i] = source[i];
            destination[2*i + 1] = 0.0;
        }else{
            destination[i] = source[2*i];
        }
        i += num_threads * gridDim.x;
    }
}

/*
 * Launch this with one block per line.
 */
template<typename scalar_type, int num_threads, int tuple_size, bool pack>
__global__ void direct_packer(int nfast, int nmid, int nslow, int line_stride, int plane_stide,
                                          scalar_type const source[], scalar_type destination[]){
    int block_index = blockIdx.x;
    while(block_index < nmid * nslow){

        int mid = block_index % nmid;
        int slow = block_index / nmid;

        scalar_type const *block_source = (pack) ?
                            &source[tuple_size * (mid * line_stride + slow * plane_stide)] :
                            &source[block_index * nfast * tuple_size];
        scalar_type *block_destination = (pack) ?
                            &destination[block_index * nfast * tuple_size] :
                            &destination[tuple_size * (mid * line_stride + slow * plane_stide)];

        int i = threadIdx.x;
        while(i < nfast * tuple_size){
            block_destination[i] = block_source[i];
            i += num_threads;
        }

        block_index += gridDim.x;
    }
}

/*
 * Call with one thread per entry.
 */
template<typename scalar_type, int num_threads>
__global__ void simple_scal(int num_entries, scalar_type data[], scalar_type scaling_factor){
    int i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        data[i] *= scaling_factor;
        i += num_threads * gridDim.x;
    }
}

/*
 * Create a 1-D CUDA thread grid using the total_threads and number of threads per block.
 * Basically, computes the number of blocks but no more than 65536.
 */
struct thread_grid_1d{
    // Compute the threads and blocks.
    thread_grid_1d(int total_threads, int num_per_block) :
        threads(num_per_block),
        blocks(std::min(total_threads / threads + ((total_threads % threads == 0) ? 0 : 1), 65536))
    {}
    // number of threads
    int const threads;
    // number of blocks
    int const blocks;
};

// max number of cuda threads (Volta supports more, but I don't think it matters)
constexpr int max_threads  = 1024;
// allows expressive calls to_complex or not to_complex
constexpr bool to_complex  = true;
// allows expressive calls to_pack or not to_pack
constexpr bool to_pack     = true;

template<typename precision_type>
void convert(int num_entries, precision_type const source[], std::complex<precision_type> destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, to_complex><<<grid.blocks, grid.threads>>>(num_entries, source, reinterpret_cast<precision_type*>(destination));
}
template<typename precision_type>
void convert(int num_entries, std::complex<precision_type> const source[], precision_type destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, not to_complex><<<grid.blocks, grid.threads>>>(num_entries, reinterpret_cast<precision_type const*>(source), destination);
}

template void convert<float>(int num_entries, float const source[], std::complex<float> destination[]);
template void convert<double>(int num_entries, double const source[], std::complex<double> destination[]);
template void convert<float>(int num_entries, std::complex<float> const source[], float destination[]);
template void convert<double>(int num_entries, std::complex<double> const source[], double destination[]);

/*
 * For float and double, defines type = <float/double> and tuple_size = 1
 * For complex float/double, defines type <float/double> and typle_size = 2
 */
template<typename scalar_type> struct precision{
    using type = scalar_type;
    static const int tuple_size = 1;
};
template<typename precision_type> struct precision<std::complex<precision_type>>{
    using type = precision_type;
    static const int tuple_size = 2;
};

template<typename scalar_type>
void direct_pack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]){
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, to_pack>
            <<<std::min(nmid * nslow, 65536), max_threads>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template<typename scalar_type>
void direct_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]){
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, not to_pack>
            <<<std::min(nmid * nslow, 65536), max_threads>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template void direct_pack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_pack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_pack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_pack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template void direct_unpack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_unpack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_unpack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_unpack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template<typename scalar_type>
void scale_data(int num_entries, scalar_type *data, double scale_factor){
    thread_grid_1d grid(num_entries, max_threads);
    simple_scal<scalar_type, max_threads><<<grid.blocks, grid.threads>>>(num_entries, data, static_cast<scalar_type>(scale_factor));
}

template void scale_data(int num_entries, float *data, double scale_factor);
template void scale_data(int num_entries, double *data, double scale_factor);

} // namespace cuda
} // namespace heffte
