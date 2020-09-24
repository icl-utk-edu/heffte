/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_backend_cuda.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#ifdef Heffte_ENABLE_MAGMA
#include <cublas.h>
#include "magma_v2.h"
#endif

namespace heffte {

namespace cuda {
void* memory_manager::allocate(size_t num_bytes){
    void *new_data;
    check_error(cudaMalloc(&new_data, num_bytes), "cudaMalloc()");
    return new_data;
}
void memory_manager::free(void *pntr){
    if (pntr != nullptr)
        check_error(cudaFree(pntr), "cudaFree()");
}
void memory_manager::host_to_device(void const *source, size_t num_bytes, void *destination){
    check_error(cudaMemcpy(destination, source, num_bytes, cudaMemcpyHostToDevice), "host_to_device (cuda)");
}
void memory_manager::device_to_device(void const *source, size_t num_bytes, void *destination){
    check_error(cudaMemcpy(destination, source, num_bytes, cudaMemcpyDeviceToDevice), "device_to_device (cuda)");
}
void memory_manager::device_to_host(void const *source, size_t num_bytes, void *destination){
    check_error(cudaMemcpy(destination, source, num_bytes, cudaMemcpyDeviceToHost), "device_to_host (cuda)");
}
}

namespace gpu {

void device_set(int active_device){
    if (active_device < 0 or active_device > device_count())
        throw std::runtime_error("device_set() called with invalid cuda device id");
    cuda::check_error(cudaSetDevice(active_device), "cudaSetDevice()");
}

void synchronize_default_stream(){
    cuda::check_error(cudaStreamSynchronize(nullptr), "device synch"); // synch the default stream
}

int device_count(){
    int count;
    cuda::check_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount()" );
    return count;
}

}

namespace cuda {

void check_error(cudaError_t status, std::string const &function_name){
    if (status != cudaSuccess)
        throw std::runtime_error(function_name + " failed with message: " + cudaGetErrorString(status));
}

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
 * Launch this with one block per line of the destination.
 */
template<typename scalar_type, int num_threads, int tuple_size, int map0, int map1, int map2>
__global__ void transpose_unpacker(int nfast, int nmid, int nslow, int line_stride, int plane_stide,
                                   int buff_line_stride, int buff_plane_stride,
                                   scalar_type const source[], scalar_type destination[]){

    int block_index = blockIdx.x;
    while(block_index < nmid * nslow){

        int j = block_index % nmid;
        int k = block_index / nmid;

        int i = threadIdx.x;
        while(i < nfast){
            if (map0 == 0 and map1 == 1 and map2 == 2){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i) + 1];
            }else if (map0 == 0 and map1 == 2 and map2 == 1){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i) + 1];
            }else if (map0 == 1 and map1 == 0 and map2 == 2){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j) + 1];
            }else if (map0 == 1 and map1 == 2 and map2 == 0){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j) + 1];
            }else if (map0 == 2 and map1 == 1 and map2 == 0){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k) + 1];
            }else if (map0 == 2 and map1 == 0 and map2 == 1){
                destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k)];
                if (tuple_size > 1)
                    destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k) + 1];
            }
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

template<typename scalar_type>
void transpose_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stride,
                      int buff_line_stride, int buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]){
    using prec = typename precision<scalar_type>::type;
    if (map0 == 0 and map1 == 1 and map2 == 2){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 0, 1, 2>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 0 and map1 == 2 and map2 == 1){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 0, 2, 1>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 1 and map1 == 0 and map2 == 2){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 1, 0, 2>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 1 and map1 == 2 and map2 == 0){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 1, 2, 0>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 2 and map1 == 0 and map2 == 1){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 2, 0, 1>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }else if (map0 == 2 and map1 == 1 and map2 == 0){
        transpose_unpacker<prec, max_threads, precision<scalar_type>::tuple_size, 2, 1, 0>
                <<<std::min(nmid * nslow, 65536), max_threads>>>
                (nfast, nmid, nslow, line_stride, plane_stride, buff_line_stride, buff_plane_stride,
                 reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
    }
}

template void direct_pack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_pack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_pack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_pack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template void direct_unpack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_unpack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_unpack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_unpack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template void transpose_unpack<float>(int, int, int, int, int, int, int, int, int, int, float const source[], float destination[]);
template void transpose_unpack<double>(int, int, int, int, int, int, int, int, int, int, double const source[], double destination[]);
template void transpose_unpack<std::complex<float>>(int, int, int, int, int, int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void transpose_unpack<std::complex<double>>(int, int, int, int, int, int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template<typename scalar_type>
void scale_data(int num_entries, scalar_type *data, double scale_factor){
    thread_grid_1d grid(num_entries, max_threads);
    simple_scal<scalar_type, max_threads><<<grid.blocks, grid.threads>>>(num_entries, data, static_cast<scalar_type>(scale_factor));
}

template void scale_data(int num_entries, float *data, double scale_factor);
template void scale_data(int num_entries, double *data, double scale_factor);

} // namespace cuda

template<typename scalar_type>
void data_manipulator<tag::gpu>::copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]){
    cuda::check_error(cudaMemcpy(destination, source, num_entries * sizeof(scalar_type), cudaMemcpyDeviceToDevice), "data_manipulator::copy_n()");
}

template void data_manipulator<tag::gpu>::copy_n<float>(float const[], size_t, float[]);
template void data_manipulator<tag::gpu>::copy_n<double>(double const[], size_t, double[]);
template void data_manipulator<tag::gpu>::copy_n<std::complex<float>>(std::complex<float> const[], size_t, std::complex<float>[]);
template void data_manipulator<tag::gpu>::copy_n<std::complex<double>>(std::complex<double> const[], size_t, std::complex<double>[]);


} // namespace heffte
