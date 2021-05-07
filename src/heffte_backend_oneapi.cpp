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

namespace oapi {

sycl::queue internal_sycl_queue = make_sycl_queue();

}

namespace gpu {

int device_count(){
    return 1;
}

void device_set(int){
    // TODO later
}

void synchronize_default_stream(){
    // OneAPI sync has to be done differently on per-queue basis
    // this syncs the internal heFFTe queue
    heffte::oapi::internal_sycl_queue.wait();
}

}

namespace oapi{

template<typename precision_type, typename index>
void convert(sycl::queue &stream, index num_entries, precision_type const source[], std::complex<precision_type> destination[]){
    precision_type* real_dest = reinterpret_cast<precision_type*>(destination);
    stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_convert_to_complex_kernel>(sycl::range<1>{static_cast<size_t>(num_entries),},
                                                                   [=](sycl::id<1> i){
                real_dest[2*i[0]] = source[i[0]];
                real_dest[2*i[0]+1] = 0.0;
            });
        }).wait();
}

template<typename precision_type, typename index>
void convert(sycl::queue &stream, index num_entries, std::complex<precision_type> const source[], precision_type destination[]){
    precision_type const* real_src = reinterpret_cast<precision_type const*>(source);
    stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_convert_to_real_kernel>(sycl::range<1>{static_cast<size_t>(num_entries),},
                                                                [=](sycl::id<1> i){
                destination[i[0]] = real_src[2*i[0]];
            });
        }).wait();
}

#define heffte_instantiate_convert(precision, index) \
    template void convert<precision, index>(sycl::queue&, index, precision const[], std::complex<precision>[]); \
    template void convert<precision, index>(sycl::queue&, index, std::complex<precision> const[], precision[]); \

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
void direct_pack(sycl::queue &stream, index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                 scalar_type const source[], scalar_type destination[]){

    stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_direct_pack_kernel>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[i[0] * nmid * nfast + i[1] * nfast + i[2]]
                        = source[ i[0] * plane_stride + i[1] * line_stride + i[2] ];
                });
    }).wait();
}

template<typename scalar_type, typename index>
void direct_unpack(sycl::queue &stream, index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                   scalar_type const source[], scalar_type destination[]){

    stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_direct_pack_kernel>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * nmid * nfast + i[1] * nfast + i[2]];
                });
    }).wait();
}

template<typename scalar_type, typename index>
void transpose_unpack(sycl::queue &stream, index nfast, index nmid, index nslow, index line_stride, index plane_stride,
                      index buff_line_stride, index buff_plane_stride, int map0, int map1, int map2,
                      scalar_type const source[], scalar_type destination[]){

    if (map0 == 0 and map1 == 1){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel012>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * buff_plane_stride + i[1] * buff_line_stride + i[2]];
                });
        }).wait();
    }else if (map0 == 0 and map1 == 2){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel021>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[1] * buff_plane_stride + i[0] * buff_line_stride + i[2]];
                });
        }).wait();
    }else if (map0 == 1 and map1 == 0){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel102>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[0] * buff_plane_stride + i[2] * buff_line_stride + i[1]];
                });
        }).wait();
    }else if (map0 == 1 and map1 == 2){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel120>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[2] * buff_plane_stride + i[0] * buff_line_stride + i[1]];
                });
        }).wait();
    }else if (map0 == 2 and map1 == 0){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel201>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[1] * buff_plane_stride + i[2] * buff_line_stride + i[0]];
                });
        }).wait();
    }else if (map0 == 2 and map1 == 1){
        stream.submit([&](sycl::handler& h){
            h.parallel_for<class heffte_transpose_unpack_kernel210>(
                sycl::range<3>{static_cast<size_t>(nslow),static_cast<size_t>(nmid),static_cast<size_t>(nfast),},
                [=](sycl::id<3> i){
                    destination[ i[0] * plane_stride + i[1] * line_stride + i[2] ]
                        = source[i[2] * buff_plane_stride + i[1] * buff_line_stride + i[0]];
                });
        }).wait();
    }else{
        throw std::runtime_error("Incorrect mapping for transpose_unpack()");
    }
}

#define heffte_instantiate_packers(index) \
template void direct_pack<float, index>(sycl::queue&, index, index, index, index, index, float const source[], float destination[]); \
template void direct_pack<double, index>(sycl::queue&, index, index, index, index, index, double const source[], double destination[]); \
template void direct_pack<std::complex<float>, index>(sycl::queue&, index, index, index, index, index, \
                                                      std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_pack<std::complex<double>, index>(sycl::queue&, index, index, index, index, index, \
                                                       std::complex<double> const source[], std::complex<double> destination[]); \
\
template void direct_unpack<float, index>(sycl::queue&, index, index, index, index, index, float const source[], float destination[]); \
template void direct_unpack<double, index>(sycl::queue&, index, index, index, index, index, double const source[], double destination[]); \
template void direct_unpack<std::complex<float>, index>(sycl::queue&, index, index, index, index, index, \
                                                        std::complex<float> const source[], std::complex<float> destination[]); \
template void direct_unpack<std::complex<double>, index>(sycl::queue&, index, index, index, index, index, \
                                                         std::complex<double> const source[], std::complex<double> destination[]); \
\
template void transpose_unpack<float, index>(sycl::queue&, index, index, index, index, index, index, index, int, int, int, \
                                             float const source[], float destination[]); \
template void transpose_unpack<double, index>(sycl::queue&, index, index, index, index, index, index, index, int, int, int, \
                                              double const source[], double destination[]); \
template void transpose_unpack<std::complex<float>, index>(sycl::queue&, index, index, index, index, index, index, index, int, int, int, \
                                                           std::complex<float> const source[], std::complex<float> destination[]); \
template void transpose_unpack<std::complex<double>, index>(sycl::queue&, index, index, index, index, index, index, index, int, int, int, \
                                                            std::complex<double> const source[], std::complex<double> destination[]); \

heffte_instantiate_packers(int)
heffte_instantiate_packers(long long)

template<typename scalar_type, typename index>
void scale_data(sycl::queue &stream, index num_entries, scalar_type *data, double scale_factor){
    stream.submit([&](sycl::handler& h){
        h.parallel_for<class heffte_scale_data_kernel>(
            sycl::range<1>{static_cast<size_t>(num_entries),}, [=](sycl::id<1> i){
                data[i[0]] *= scale_factor;
            });
    }).wait();
}

template void scale_data<float, int>(sycl::queue&, int num_entries, float *data, double scale_factor);
template void scale_data<double, int>(sycl::queue&, int num_entries, double *data, double scale_factor);
template void scale_data<float, long long>(sycl::queue&, long long num_entries, float *data, double scale_factor);
template void scale_data<double, long long>(sycl::queue&, long long num_entries, double *data, double scale_factor);

} // namespace oapi

} // namespace heffte

#endif
