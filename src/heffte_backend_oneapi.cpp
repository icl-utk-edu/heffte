/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#define HIP_ENABLE_PRINTF

#include "heffte_backend_oneapi.h"

#ifdef Heffte_ENABLE_ONEAPI

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>
#include "oneapi/mkl.hpp"

namespace heffte {

namespace oneapi {

sycl::queue* make_sycl_queue(){
    try{
        return new sycl::queue(sycl::gpu_selector());
    }catch(sycl::exception const&){
        return new sycl::queue(sycl::cpu_selector());
    }
}

heffte_internal_sycl_queue def_queue;

void* memory_manager::allocate(size_t num_bytes){
    void* result = reinterpret_cast<void*>( sycl::malloc_device<char>(num_bytes, def_queue) );
    def_queue.wait();
    return result;
}
void memory_manager::free(void *pntr){
    if (pntr != nullptr)
        sycl::free(pntr, def_queue);
    def_queue.wait();
}
void memory_manager::host_to_device(void const *source, size_t num_bytes, void *destination){
    def_queue->memcpy(destination, source, num_bytes);
    def_queue.wait();
}
void memory_manager::device_to_device(void const *source, size_t num_bytes, void *destination){
    def_queue->memcpy(destination, source, num_bytes);
    def_queue.wait();
}
void memory_manager::device_to_host(void const *source, size_t num_bytes, void *destination){
    def_queue->memcpy(destination, source, num_bytes);
    def_queue.wait();
}
}

namespace gpu {

int device_count(){
    return 1;
}

void device_set(int){
    // TODO later
}

void synchronize_default_stream(){
    // OneAPI sync has to be done differently
}

}
//
// namespace rocm {
//
// void check_error(hipError_t status, std::string const &function_name){
//     if (status != hipSuccess)
//         throw std::runtime_error(function_name + " failed with message: " + std::to_string(status));
// }
//
// /*
//  * Launch with one thread per entry.
//  *
//  * If to_complex is true, convert one real number from source to two real numbers in destination.
//  * If to_complex is false, convert two real numbers from source to one real number in destination.
//  */
// template<typename scalar_type, int num_threads, bool to_complex, typename index>
// __global__ __launch_bounds__(num_threads) void real_complex_convert(index num_entries, scalar_type const source[], scalar_type destination[]){
//     index i = blockIdx.x * num_threads + threadIdx.x;
//     while(i < num_entries){
//         if (to_complex){
//             destination[2*i] = source[i];
//             destination[2*i + 1] = 0.0;
//         }else{
//             destination[i] = source[2*i];
//         }
//         i += num_threads * gridDim.x;
//     }
// }
//
// /*
//  * Launch this with one block per line.
//  */
// template<typename scalar_type, int num_threads, int tuple_size, bool pack, typename index>
// __global__ __launch_bounds__(num_threads) void direct_packer(index nfast, index nmid, index nslow, index line_stride, index plane_stide,
//                                           scalar_type const source[], scalar_type destination[]){
//     index block_index = blockIdx.x;
//     while(block_index < nmid * nslow){
//
//         index mid = block_index % nmid;
//         index slow = block_index / nmid;
//
//         scalar_type const *block_source = (pack) ?
//                             &source[tuple_size * (mid * line_stride + slow * plane_stide)] :
//                             &source[block_index * nfast * tuple_size];
//         scalar_type *block_destination = (pack) ?
//                             &destination[block_index * nfast * tuple_size] :
//                             &destination[tuple_size * (mid * line_stride + slow * plane_stide)];
//
//         index i = threadIdx.x;
//         while(i < nfast * tuple_size){
//             block_destination[i] = block_source[i];
//             i += num_threads;
//         }
//
//         block_index += gridDim.x;
//     }
// }
//
// /*
//  * Launch this with one block per line of the destination.
//  */
// template<typename scalar_type, int num_threads, int tuple_size, int map0, int map1, int map2, typename index>
// __global__ __launch_bounds__(num_threads) void transpose_unpacker(index nfast, index nmid, index nslow, index line_stride, index plane_stide,
//                                    index buff_line_stride, index buff_plane_stride,
//                                    scalar_type const source[], scalar_type destination[]){
//
//     index block_index = blockIdx.x;
//     while(block_index < nmid * nslow){
//
//         index j = block_index % nmid;
//         index k = block_index / nmid;
//
//         index i = threadIdx.x;
//         while(i < nfast){
//             if (map0 == 0 and map1 == 1 and map2 == 2){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + j * buff_line_stride + i) + 1];
//             }else if (map0 == 0 and map1 == 2 and map2 == 1){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + k * buff_line_stride + i) + 1];
//             }else if (map0 == 1 and map1 == 0 and map2 == 2){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (k * buff_plane_stride + i * buff_line_stride + j) + 1];
//             }else if (map0 == 1 and map1 == 2 and map2 == 0){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + k * buff_line_stride + j) + 1];
//             }else if (map0 == 2 and map1 == 1 and map2 == 0){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (i * buff_plane_stride + j * buff_line_stride + k) + 1];
//             }else if (map0 == 2 and map1 == 0 and map2 == 1){
//                 destination[tuple_size * (k * plane_stide + j * line_stride + i)] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k)];
//                 if (tuple_size > 1)
//                     destination[tuple_size * (k * plane_stide + j * line_stride + i) + 1] = source[tuple_size * (j * buff_plane_stride + i * buff_line_stride + k) + 1];
//             }
//             i += num_threads;
//         }
//
//         block_index += gridDim.x;
//     }
// }
//
// /*
//  * Call with one thread per entry.
//  */
// template<typename scalar_type, int num_threads, typename index>
// __global__ __launch_bounds__(num_threads) void simple_scal(index num_entries, scalar_type data[], scalar_type scaling_factor){
//     index i = blockIdx.x * num_threads + threadIdx.x;
//     while(i < num_entries){
//         data[i] *= scaling_factor;
//         i += num_threads * gridDim.x;
//     }
// }
//
// /*
//  * Create a 1-D CUDA thread grid using the total_threads and number of threads per block.
//  * Basically, computes the number of blocks but no more than 65536.
//  */
// struct thread_grid_1d{
//     // Compute the threads and blocks.
//     thread_grid_1d(int total_threads, int num_per_block) :
//         threads(num_per_block),
//         blocks(std::min(total_threads / threads + ((total_threads % threads == 0) ? 0 : 1), 65536))
//     {}
//     // number of threads
//     int const threads;
//     // number of blocks
//     int const blocks;
// };
//
// // max number of cuda threads (Volta supports more, but I don't think it matters)
// constexpr int max_threads  = 1024;
// // allows expressive calls to_complex or not to_complex
// constexpr bool to_complex  = true;
// // allows expressive calls to_pack or not to_pack
// constexpr bool to_pack     = true;
//

namespace oneapi{

template<typename precision_type, typename index>
void convert(index num_entries, precision_type const source[], std::complex<precision_type> destination[]){
    precision_type* real_dest = reinterpret_cast<precision_type*>(destination);
    def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_convert_to_complex_kernel>(sycl::range<1>{static_cast<size_t>(num_entries),},
                                                                   [=](sycl::id<1> i){
                real_dest[2*i[0]] = source[i[0]];
                real_dest[2*i[0]+1] = 0.0;
            });
        });
    def_queue.wait();
}

template<typename precision_type, typename index>
void convert(index num_entries, std::complex<precision_type> const source[], precision_type destination[]){
    precision_type const* real_src = reinterpret_cast<precision_type const*>(source);
    def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_convert_to_real_kernel>(sycl::range<1>{static_cast<size_t>(num_entries),},
                                                                [=](sycl::id<1> i){
                destination[i[0]] = real_src[2*i[0]];
            });
        });
    def_queue.wait();
}

#define heffte_instantiate_convert(precision, index) \
    template void convert<precision, index>(index num_entries, precision const source[], std::complex<precision> destination[]); \
    template void convert<precision, index>(index num_entries, std::complex<precision> const source[], precision destination[]); \

heffte_instantiate_convert(float, int)
heffte_instantiate_convert(double, int)
heffte_instantiate_convert(float, long long)
heffte_instantiate_convert(double, long long)

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

template<typename scalar_type, typename index>
void direct_pack(index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                 scalar_type const source[], scalar_type destination[]){

    def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_direct_pack_kernel>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[i[0] * nmid * nfast + i[1] * nfast + i[2]]
                        = source[ i[0] * plane_stride + i[1] * line_stride + i[2] ];
                });
    });
    def_queue.wait();
}

template<typename scalar_type, typename index>
void direct_unpack(index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                   scalar_type const source[], scalar_type destination[]){

    def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_direct_pack_kernel>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * nmid * nfast + i[1] * nfast + i[2]];
                });
    });
    def_queue.wait();
}

template<typename scalar_type, typename index>
void transpose_unpack(index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                      index buff_line_stride, index buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]){

    if (map0 == 0 and map1 == 1){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel012>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * buff_plane_stride + i[1] * buff_line_stride + i[2]];
                });
        });
    }else if (map0 == 0 and map1 == 2){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel021>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[1] * buff_plane_stride + i[0] * buff_line_stride + i[2]];
                });
        });
    }else if (map0 == 1 and map1 == 0){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel102>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * buff_plane_stride + i[2] * buff_line_stride + i[1]];
                });
        });
    }else if (map0 == 1 and map1 == 2){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel120>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[2] * buff_plane_stride + i[0] * buff_line_stride + i[1]];
                });
        });
    }else if (map0 == 2 and map1 == 0){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel201>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[1] * buff_plane_stride + i[2] * buff_line_stride + i[0]];
                });
        });
    }else if (map0 == 2 and map1 == 1){
        def_queue->submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel210>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[2] * buff_plane_stride + i[1] * buff_line_stride + i[0]];
                });
        });
    }else{
        throw std::runtime_error("Incorrect mapping for transpose_unpack()");
    }
    def_queue.wait();

}

#define heffte_instantiate_packers(index) \
template void direct_pack<float, index>(index, index, index, index, index, float const source[], float destination[]); \
template void direct_pack<double, index>(index, index, index, index, index, double const source[], double destination[]); \
template void direct_pack<std::complex<float>, index>(index, index, index, index, index, \
                                                      std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_pack<std::complex<double>, index>(index, index, index, index, index, \
                                                       std::complex<double> const source[], std::complex<double> destination[]); \
\
template void direct_unpack<float, index>(index, index, index, index, index, float const source[], float destination[]); \
template void direct_unpack<double, index>(index, index, index, index, index, double const source[], double destination[]); \
template void direct_unpack<std::complex<float>, index>(index, index, index, index, index, \
                                                        std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_unpack<std::complex<double>, index>(index, index, index, index, index, \
                                                         std::complex<double> const source[], std::complex<double> destination[]); \
\
template void transpose_unpack<float, index>(index, index, index, index, index, index, index, int, int, int, \
                                             float const source[], float destination[]); \
template void transpose_unpack<double, index>(index, index, index, index, index, index, index, int, int, int, \
                                              double const source[], double destination[]); \
template void transpose_unpack<std::complex<float>, index>(index, index, index, index, index, index, index, int, int, int, \
                                                           std::complex<float> const source[], std::complex<float> destination[]); \
template void transpose_unpack<std::complex<double>, index>(index, index, index, index, index, index, index, int, int, int, \
                                                            std::complex<double> const source[], std::complex<double> destination[]); \

heffte_instantiate_packers(int)
heffte_instantiate_packers(long long)

template<typename scalar_type, typename index>
void scale_data(index num_entries, scalar_type *data, double scale_factor){
    def_queue->submit([&](sycl::handler& h){
        h.parallel_for<class heffte_scale_data_kernel>(
            sycl::range<1>{static_cast<size_t>(num_entries),}, [=](sycl::id<1> i){
                data[i[0]] *= scale_factor;
            });
    });
    def_queue.wait();
}

template void scale_data<float, int>(int num_entries, float *data, double scale_factor);
template void scale_data<double, int>(int num_entries, double *data, double scale_factor);
template void scale_data<float, long long>(long long num_entries, float *data, double scale_factor);
template void scale_data<double, long long>(long long num_entries, double *data, double scale_factor);

} // namespace oneapi

template<typename scalar_type>
void data_manipulator<tag::gpu>::copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]){
    oneapi::def_queue->memcpy(destination, source, num_entries * sizeof(scalar_type));
    oneapi::def_queue.wait();
}

template void data_manipulator<tag::gpu>::copy_n<float>(float const[], size_t, float[]);
template void data_manipulator<tag::gpu>::copy_n<double>(double const[], size_t, double[]);
template void data_manipulator<tag::gpu>::copy_n<std::complex<float>>(std::complex<float> const[], size_t, std::complex<float>[]);
template void data_manipulator<tag::gpu>::copy_n<std::complex<double>>(std::complex<double> const[], size_t, std::complex<double>[]);


} // namespace heffte

#endif
