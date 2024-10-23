/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include <random>

#include "test_common.h"

void test_factorize(){
    current_test<int, using_nompi> name("prime factorize");

    std::vector<std::array<int, 2>> reference = {{1, 935}, {5, 187}, {11, 85}, {17, 55}, {55, 17}, {85, 11}, {187, 5}, {935, 1}};

    auto factors = heffte::get_factors(935);

    sassert(match(factors, reference));

    reference = {{1, 27}, {3, 9}, {9, 3}, {27, 1}};
    factors = heffte::get_factors(reference.front()[1]);
    sassert(match(factors, reference));
}

void test_process_grid(){
    current_test<int, using_nompi> name("process grid");

    std::array<int, 2> reference = {4, 5};
    std::array<int, 2> result = heffte::make_procgrid(20);
    sassert(reference == result);

    reference = {1, 17};
    result = heffte::make_procgrid(17);
    sassert(reference == result);

    reference = {81, 81};
    result = heffte::make_procgrid(6561);
    sassert(reference == result);

    reference = {17, 19};
    result = heffte::make_procgrid(323);
    sassert(reference == result);

    reference = {8, 16};
    result = heffte::make_procgrid(128);
    sassert(reference == result);
}

void test_split_pencils(){
    using namespace heffte;
    current_test<int, using_nompi> name("split pencils");

    box3d<> const world = {{0, 0, 0}, {1, 3, 5}};
    std::vector<box3d<>> reference = {{{0, 0, 0}, {0, 1, 5}}, {{0, 2, 0}, {0, 3, 5}},
                                      {{1, 0, 0}, {1, 1, 5}}, {{1, 2, 0}, {1, 3, 5}}};
    // note that the order of the boxes moves fastest in the mid-dimension
    // this tests the reordering
    std::vector<box3d<>> result = make_pencils(world, {2, 2}, 2, reference, world.order);
    sassert(match(result, reference));

    std::vector<box3d<>> reference2 = {{{0, 0, 0}, {1, 1, 2}}, {{0, 2, 0}, {1, 3, 2}},
                                       {{0, 0, 3}, {1, 1, 5}}, {{0, 2, 3}, {1, 3, 5}}};
    std::vector<box3d<>> result2 = make_pencils(world, {2, 2}, 0, reference, world.order);
    sassert(match(result2, reference2));

    box3d<> const reconstructed_world = find_world(result);
    sassert(reconstructed_world == world);
}

void test_cpu_scale(){
    using namespace heffte;
    current_test<int, using_nompi> name("cpu scaling");
    std::vector<float> x = {1.0, 33.0, 88.0, -11.0, 2.0};
    std::vector<float> y = x;
    for(auto &v : y) v *= 3.0;
    data_scaling::apply(x.size(), x.data(), 3.0);
    sassert(approx(x, y));

    std::vector<std::complex<double>> cx = {{1.0, -11.0}, {33.0, 8.0}, {88.0, -11.0}, {2.0, -9.0}};
    std::vector<std::complex<double>> cy = cx;
    for(auto &v : cy) v /= 1.33;
    data_scaling::apply(cx.size(), cx.data(), 1.0 / 1.33);
    sassert(approx(cx, cy));
}

/*
 * Generates input for the fft, the input consists of reals or complex but they have only
 * integer values and the values follow the order of the entries.
 * Designed to work on a grid of size {2, 3, 4} for total of 24 entries.
 */
template<typename scalar_type>
std::vector<scalar_type> make_input(){
    std::vector<scalar_type> result(24);
    for(int i=0; i<24; i++) result[i] = static_cast<scalar_type>(i + 1);
    return result;
}

/*
 * Reorder a set of indexes with given size into the given order,
 * the input order is assumed the canonical (0, 1, 2).
 */
template<typename scalar_type>
std::vector<scalar_type> reorder_box(std::array<int, 3> size, std::array<int, 3> order, std::vector<scalar_type> const &input){
    std::vector<scalar_type> result(input.size());

    std::array<int, 3> max_iter = {size[order[0]], size[order[1]], size[order[2]]};
    std::array<int, 3> iter = {0, 0, 0};
    int &ti = (order[0] == 0) ? iter[0] : ((order[1] == 0) ? iter[1] : iter[2]);
    int &tj = (order[0] == 1) ? iter[0] : ((order[1] == 1) ? iter[1] : iter[2]);
    int &tk = (order[0] == 2) ? iter[0] : ((order[1] == 2) ? iter[1] : iter[2]);

    int plane = size[order[0]] * size[order[1]];
    int lane  = size[order[0]];

    int oplane = size[0] * size[1];
    int olane = size[0];

    for(iter[2]=0; iter[2]<max_iter[2]; iter[2]++){
        for(iter[1]=0; iter[1]<max_iter[1]; iter[1]++){
            for(iter[0]=0; iter[0]<max_iter[0]; iter[0]++){
                result[iter[2] * plane + iter[1] * lane + iter[0]] = input[tk * oplane + tj * olane + ti];
            }
        }
    }

    return result;
}

/*
 * Considering the input generated by the make_input() method and box {{0, 0, 0}, {1, 2, 3}}
 * constructs the corresponding fft coefficients assuming 1-D transforms have
 * been applied across the zeroth dimension.
 * Each transform uses 2 entries, since the size in dimension 0 is 2.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft0(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<result.size(); i+=2){
        result[i]   = static_cast<typename fft_output<scalar_type>::type>(3 + 2 * i);
        result[i+1] = static_cast<typename fft_output<scalar_type>::type>(-1.0);
    }
    return result;
}
/*
 * Same as make_fft0() but the transforms are applied to dimension 1.
 * Each transform uses 3 entries, since the size in dimension 1 is 3.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft1(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(int j=0; j<4; j++){
        for(int i=0; i<2; i++){
            result[6 * j + i]     = typename fft_output<scalar_type>::type((2*j + i+1) * 9.0 - i * 6.0);
            result[6 * j + i + 2] = typename fft_output<scalar_type>::type(-3.0,  1.73205080756888);
            result[6 * j + i + 4] = typename fft_output<scalar_type>::type(-3.0, -1.73205080756888);
        }
    }
    return result;
}
/*
 * Same as make_fft1() but using the r2c transform and only the unique entries.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft1_r2c(){
    std::vector<typename fft_output<scalar_type>::type> result(16);
    for(int j=0; j<4; j++){
        for(int i=0; i<2; i++){
            result[4 * j + i]     = typename fft_output<scalar_type>::type((2*j + i+1) * 9.0 - i * 6.0);
            result[4 * j + i + 2] = typename fft_output<scalar_type>::type(-3.0,  1.73205080756888);
        }
    }
    return result;
}
/*
 * Same as make_fft0() but the transforms are applied to dimension 2.
 * Each transform uses 4 entries, since the size in dimension 2 is 4.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft2(){
    std::vector<typename fft_output<scalar_type>::type> result(24);
    for(size_t i=0; i<6; i++){
        result[i]    = typename fft_output<scalar_type>::type(40.0 + 4 * i);
        result[i+ 6] = typename fft_output<scalar_type>::type(-12.0, 12.0);
        result[i+12] = typename fft_output<scalar_type>::type(-12.0);
        result[i+18] = typename fft_output<scalar_type>::type(-12.0, -12.0);
    }
    return result;
}
/*
 * Same as make_fft2() but using the r2c transform and only the unique entries.
 */
template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> make_fft2_r2c(){
    std::vector<typename fft_output<scalar_type>::type> result(18);
    for(size_t i=0; i<6; i++){
        result[i]    = typename fft_output<scalar_type>::type(40.0 + 4 * i);
        result[i+ 6] = typename fft_output<scalar_type>::type(-12.0, 12.0);
        result[i+12] = typename fft_output<scalar_type>::type(-12.0);
    }
    return result;
}

/*
 * Tests the 1-D executor for the backend_tag in the complex-to-complex case.
 * The test is done against pen-and-paper solution using a box of size 2x3x4 with data 1, 2, 3, ...
 * The reference solution vectors are generated by make_fft0/1/2.
 */
template<typename backend_tag, typename scalar_type>
void test_1d_complex(){
    current_test<scalar_type, using_nompi> name(backend::name<backend_tag>() + " one-dimension");

    box3d<> const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with make_input and make_fft methods

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };
    backend::device_instance<typename backend::buffer_traits<backend_tag>::location> device;

    for(size_t i=0; i<reference.size(); i++){
        auto fft = heffte::make_executor<backend_tag>(device.stream(), box, i);
        auto workspace = make_buffer_container<typename fft_output<scalar_type>::type>(device.stream(), fft->workspace_size());

        auto forward_result = test_traits<backend_tag>::load(input);
        fft->forward(forward_result.data(), workspace.data());
        sassert(approx(forward_result, reference[i]));

        fft->backward(forward_result.data(), workspace.data());

        auto backward_result = test_traits<backend_tag>::unload(forward_result);
        for(auto &r : backward_result) r /= (2.0 + i);
        sassert(approx(backward_result, input));
    }
}
// Same as test_1d_complex() but uses the real-to-complex case computing all entries.
template<typename backend_tag, typename scalar_type>
void test_1d_real(){
    current_test<scalar_type, using_nompi> name(backend::name<backend_tag>() + " one-dimension");

    box3d<> const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1<scalar_type>(), make_fft2<scalar_type>() };
    backend::device_instance<typename backend::buffer_traits<backend_tag>::location> device;

    for(size_t i=0; i<reference.size(); i++){
        auto fft = heffte::make_executor<backend_tag>(device.stream(), box, i);
        auto workspace = heffte::make_buffer_container<std::complex<scalar_type>>(device.stream(), fft->workspace_size());

        auto load_input = test_traits<backend_tag>::load(input);
        typename test_traits<backend_tag>::template container<typename fft_output<scalar_type>::type> result(input.size());
        fft->forward(load_input.data(), result.data(), workspace.data());
        sassert(approx(result, reference[i]));

        typename test_traits<backend_tag>::template container<scalar_type> back_result(result.size());
        fft->backward(result.data(), back_result.data(), workspace.data());
        auto unload_result = test_traits<backend_tag>::unload(back_result);
        for(auto &r : unload_result) r /= (2.0 + i);
        sassert(approx(unload_result, input));
    }
}
template<typename location_tag, typename scalar_type>
void test_real_rule(typename backend::device_instance<location_tag>::stream_type stream,
                    std::unique_ptr<executor_base> const &fft, size_t batch_size,
                    std::vector<scalar_type> const &input, std::vector<scalar_type> const &reference,
                    double scale_factor){

    std::vector<scalar_type> full_input(batch_size * input.size());
    for(size_t i=0; i<batch_size; i++) std::copy(input.begin(), input.end(), full_input.begin() + i * input.size());
    std::vector<scalar_type> full_reference(batch_size * reference.size());
    for(size_t i=0; i<batch_size; i++) std::copy(reference.begin(), reference.end(), full_reference.begin() + i * reference.size());

    auto workspace = make_buffer_container<scalar_type>(stream, fft->workspace_size());

    auto load_input = test_traits<location_tag>::load(full_input);
    fft->forward(load_input.data(), workspace.data());
    sassert(approx(load_input, full_reference));

    fft->backward(load_input.data(), workspace.data());
    auto unload_result = test_traits<location_tag>::unload(load_input);
    for(auto &r : unload_result) r *= scale_factor;
    sassert(approx(unload_result, full_input));
}

// Same as test_1d_complex() but uses the real-to-complex case computing all entries.
template<typename cos_tag, typename sin_tag, typename scalar_type>
void test_1d_r2r(){
    using location_tag = typename backend::buffer_traits<cos_tag>::location;

    static_assert(std::is_same<location_tag, typename backend::buffer_traits<sin_tag>::location>::value,
                  "Cannot mix incompatible sin-cos backends.");

    box3d<> const box = {{0, 0, 0}, {3, 1, 2}};
    box3d<> const box5 = {{0, 0, 0}, {4, 0, 2}};

    backend::device_instance<typename backend::buffer_traits<cos_tag>::location> device;

    scalar_type box_scale = 1.0 / static_cast<scalar_type>(4 * box.size[0]);
    scalar_type box5_scale = 1.0 / static_cast<scalar_type>(4 * box5.size[0]);
    if (std::is_same<cos_tag, backend::fftw_cos>::value or std::is_same<cos_tag, backend::fftw_sin>::value) {
        box_scale = 1.0 / static_cast<scalar_type>(2 * box.size[0]);
        box5_scale = 1.0 / static_cast<scalar_type>(2 * box5.size[0]);
    }

    {
    current_test<scalar_type, using_nompi> name(backend::name<cos_tag>() + " one-dimension");
    test_real_rule<location_tag, scalar_type>(device.stream(),
                                              heffte::make_executor<cos_tag>(device.stream(), box, 0), 6,
                                              std::vector<scalar_type>{1.0, 2.0, 3.0, 4.0},
                                              std::vector<scalar_type>{2.0000000000000000e+01, -6.3086440597978992e+00, 0.0000000000000000e+00, -4.4834152916796510e-01},
                                              box_scale);

    test_real_rule<location_tag, scalar_type>(device.stream(),
                                              heffte::make_executor<cos_tag>(device.stream(), box5, 0), 3,
                                              std::vector<scalar_type>{1.0, 2.0, 3.0, 4.0, 5.0},
                                              std::vector<scalar_type>{30.0, -9.9595931395311208, 0.0, -8.9805595315917053e-01, 0.0},
                                              box5_scale);
    }{
    current_test<scalar_type, using_nompi> name(backend::name<sin_tag>() + " one-dimension");
    test_real_rule<location_tag, scalar_type>(device.stream(),
                                              heffte::make_executor<sin_tag>(device.stream(), box, 0), 6,
                                              std::vector<scalar_type>{1.0, 2.0, 3.0, 4.0},
                                              std::vector<scalar_type>{1.3065629648763766e+01, -5.6568542494923806e+00, 5.4119610014619699e+00, -4.0e+00},
                                              box_scale);

    test_real_rule<location_tag, scalar_type>(device.stream(),
                                              heffte::make_executor<sin_tag>(device.stream(), box5, 0), 3,
                                              std::vector<scalar_type>{1.0, 2.0, 3.0, 4.0, 5.0},
                                              std::vector<scalar_type>{1.9416407864998735e+01, -8.5065080835203979e+00, 7.4164078649987371e+00, -5.2573111211913348e+00, 6.0e+00},
                                              box5_scale);
    }
}

// Same as test_1d_complex() but uses the r2c case computing only the non-conjugate complex entries.
template<typename backend_tag, typename scalar_type>
void test_1d_r2c(){
    current_test<scalar_type, using_nompi> name(backend::name<backend_tag>() + " one-dimension r2c");

    box3d<> const box = {{0, 0, 0}, {1, 2, 3}}; // sync this with the "answers" vector

    auto const input = make_input<scalar_type>();
    std::vector<std::vector<typename fft_output<scalar_type>::type>> reference =
        { make_fft0<scalar_type>(), make_fft1_r2c<scalar_type>(), make_fft2_r2c<scalar_type>() };
    backend::device_instance<typename backend::buffer_traits<backend_tag>::location> device;

    #ifdef Heffte_ENABLE_ROCM
    if (std::is_same<backend_tag, backend::rocfft>::value)
        reference.resize(1); // the rocFFT strided transforms are not supported yet, test only the contiguous one corresponding to i = 0
    #endif

    for(size_t i=0; i<reference.size(); i++){
        auto fft = heffte::make_executor_r2c<backend_tag>(device.stream(), box, i);
        auto workspace = make_buffer_container<typename fft_output<scalar_type>::type>(device.stream(), fft->workspace_size());

        auto load_input = test_traits<backend_tag>::load(input);
        typename test_traits<backend_tag>::template container<typename fft_output<scalar_type>::type> result(fft->complex_size());
        fft->forward(load_input.data(), result.data(), workspace.data());
        sassert(approx(result, reference[i]));

        typename test_traits<backend_tag>::template container<scalar_type> back_result(input.size());
        fft->backward(result.data(), back_result.data(), workspace.data());
        auto unload_result = test_traits<backend_tag>::unload(back_result);
        for(auto &r : unload_result) r /= (2.0 + i);
        sassert(approx(unload_result, input));
    }
}
// instantiates a test for the backend tag using all cases of 1D real and complex transforms
template<typename backend_tag>
void test_1d(){
    test_1d_real<backend_tag, float>();
    test_1d_real<backend_tag, double>();
    test_1d_complex<backend_tag, std::complex<float>>();
    test_1d_complex<backend_tag, std::complex<double>>();
    test_1d_r2c<backend_tag, float>();
    test_1d_r2c<backend_tag, double>();
}
template<typename cos_tag, typename sin_tag>
void test_1d_r2r(){
    test_1d_r2r<cos_tag, sin_tag, float>();
    test_1d_r2r<cos_tag, sin_tag, double>();
}

void test_1d_all(){
    test_1d<backend::stock>();
    test_1d_r2r<backend::stock_cos, backend::stock_sin>();
    #ifdef Heffte_ENABLE_FFTW
    test_1d<backend::fftw>();
    test_1d_r2r<backend::fftw_cos, backend::fftw_sin>();
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_1d<backend::mkl>();
    test_1d_r2r<backend::mkl_cos, backend::mkl_sin>();
    #endif
    #ifdef Heffte_ENABLE_GPU
    test_1d<gpu_backend>(); // pick the default GPU backend
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_1d_r2r<backend::cufft_cos, backend::cufft_sin>();
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_1d_r2r<backend::rocfft_cos, backend::rocfft_sin>();
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_1d_r2r<backend::onemkl_cos, backend::onemkl_sin>();
    #endif
}

#ifdef Heffte_ENABLE_GPU
// Tests the load/unload methods associated with the gpu vector template.
template<typename scalar_type>
void test_gpu_vector(size_t num_entries){
    static_assert(std::is_copy_constructible<gpu::vector<scalar_type>>::value, "Lost copy-constructor for gpu::vector");
    static_assert(std::is_move_constructible<gpu::vector<scalar_type>>::value, "Lost move-constructor for gpu::vector");
    static_assert(std::is_copy_assignable<gpu::vector<scalar_type>>::value, "Lost copy-assign for gpu::vector");
    static_assert(std::is_move_assignable<gpu::vector<scalar_type>>::value, "Lost move-assign for gpu::vector");

    current_test<scalar_type, using_nompi> name("gpu::vector");
    std::vector<scalar_type> source(num_entries);
    std::iota(source.begin(), source.end(), 0); // fill source with 0, 1, 2, 3, 4 ...
    gpu::vector<scalar_type> v1 = gpu::transfer::load(source);
    sassert(v1.size() == source.size());
    gpu::vector<scalar_type> v2 = v1; // test copy constructor
    sassert(v1.size() == v2.size());
    std::vector<scalar_type> dest = gpu::transfer::unload(v2);
    sassert(match(dest, source));

    { // test move constructor
        gpu::vector<scalar_type> t = std::move(v2);
        dest = std::vector<scalar_type>(); // reset the destination
        dest = gpu::transfer::unload(t);
        sassert(match(dest, source));
    }

    sassert(v2.empty()); // test empty and reset to null after move
    v2 = std::move(v1);  // test move assignment
    sassert(v1.empty()); // test if moved output_forward of v1

    dest = std::vector<scalar_type>(); // reset the destination
    dest = gpu::transfer::unload(v2);
    sassert(match(dest, source));

    v1 = gpu::transfer::load(source);
    v2 = gpu::vector<scalar_type>(v1.data(), v1.data() + num_entries / 2);
    sassert(v2.size() == num_entries / 2);
    dest = gpu::transfer::unload(v2);
    source.resize(num_entries / 2);
    sassert(match(dest, source));

    size_t num_v2 = v2.size();
    scalar_type *raw_array = v2.release();
    sassert(v2.empty());
    v2 = gpu::transfer::capture(std::move(raw_array), num_v2);
    sassert(raw_array == nullptr);
    sassert(not v2.empty());
}
void test_gpu_vector(){
    test_gpu_vector<float>(11);
    test_gpu_vector<double>(40);
    test_gpu_vector<std::complex<float>>(73);
    test_gpu_vector<std::complex<double>>(13);
}
// Tests the gpu scaling kernel.
void test_gpu_scale(){
    using namespace heffte;
    current_test<int, using_nompi> name("gpu scaling");
    std::vector<float> x = {1.0, 33.0, 88.0, -11.0, 2.0};
    std::vector<float> y = x;
    for(auto &v : y) v *= 3.0;
    auto gx = gpu::transfer::load(x);
    backend::device_instance<tag::gpu> device;
    data_scaling::apply(device.stream(), gx.size(), gx.data(), 3.0);
    x = gpu::transfer::unload(gx);
    sassert(approx(x, y));

    std::vector<std::complex<double>> cx = {{1.0, -11.0}, {33.0, 8.0}, {88.0, -11.0}, {2.0, -9.0}};
    std::vector<std::complex<double>> cy = cx;
    for(auto &v : cy) v /= 1.33;
    auto gcx = gpu::transfer::load(cx);
    data_scaling::apply(device.stream(), gcx.size(), gcx.data(), 1.0 / 1.33);
    cx = gpu::transfer::unload(gcx);
    sassert(approx(cx, cy));
}
#else
void test_gpu_vector(){} // empty methods to call in case there is no GPU backend
void test_gpu_scale(){}
#endif

template<typename backend_tag>
void test_1d_reorder_one(box3d<int> const box,
                         std::vector<double> const &rinput, std::vector<std::complex<double>> const &cinput,
                         std::vector<std::vector<std::complex<double>>> const &rreference,
                         std::vector<std::vector<std::complex<double>>> const &creference){

    using location_tag = typename backend::buffer_traits<backend_tag>::location;

    backend::device_instance<location_tag> device;

    for(size_t i=0; i<3; i++){
        auto fft = make_executor<backend_tag>(device.stream(), box, box.order[i]);
        auto fft_r2c = make_executor_r2c<backend_tag>(device.stream(), box, box.order[i]);

        auto cresult = test_traits<location_tag>::load(cinput);
        auto workspace = make_buffer_container<std::complex<double>>(device.stream(),
                                                                     std::max(fft->workspace_size(), fft_r2c->workspace_size()));
        auto rresult = make_buffer_container<std::complex<double>>(device.stream(), rreference[i].size());
        auto bresult = make_buffer_container<double>(device.stream(), rinput.size());

        fft->forward(cresult.data(), workspace.data());
        sassert(approx(cresult, creference[i]));

        fft->backward(cresult.data(), workspace.data());
        auto cpu_cresult = test_traits<location_tag>::unload(cresult);
        for(auto &r : cpu_cresult) r /= (2.0 + box.order[i]);
        sassert(approx(cpu_cresult, cinput));

        // rocFFT r2c does not support strided transforms
        if (std::is_same<backend_tag, backend::rocfft>::value) continue;

        auto loaded_input = test_traits<location_tag>::load(rinput);
        fft_r2c->forward(loaded_input.data(), rresult.data(), workspace.data());
        sassert(approx(rresult, rreference[i]));

        fft_r2c->backward(rresult.data(), bresult.data(), workspace.data());
        auto cpu_rresult = test_traits<location_tag>::unload(bresult);
        for(auto &r : cpu_rresult) r /= (2.0 + box.order[i]);
        sassert(approx(cpu_rresult, rinput));
    }
}

/*
 * Mostly the same as the other 1D case testing against the pen-and-paper solution,
 * but this one uses a box with reordered entries.
 */
void test_1d_reorder(){
    // includes both c2c and r2c cases
    using rtype = double;
    using ctype = std::complex<double>;
    current_test<rtype, using_nompi> name("one-dimension reorder logic");

    // make a box
    box3d<> const box({0, 0, 0}, {1, 2, 3}, {2, 0, 1}); // sync this with make_input and make_fft methods

    auto const rinput = reorder_box(box.size, box.order, make_input<rtype>());
    auto const cinput = reorder_box(box.size, box.order, make_input<ctype>());

    std::vector<std::vector<ctype>> rreference =
        { make_fft2_r2c<ctype>(), make_fft0<ctype>(), make_fft1_r2c<ctype>() };
    std::vector<std::vector<ctype>> creference =
        { make_fft2<ctype>(), make_fft0<ctype>(), make_fft1<ctype>() };

    for(size_t i=0; i<3; i++){
        rreference[i] = reorder_box(box.r2c(box.order[i]).size, box.order, rreference[i]);
        creference[i] = reorder_box(box.size, box.order, creference[i]);
    }

    test_1d_reorder_one<backend::stock>(box, rinput, cinput, rreference, creference);

    #ifdef Heffte_ENABLE_FFTW
    test_1d_reorder_one<backend::fftw>(box, rinput, cinput, rreference, creference);
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_1d_reorder_one<backend::mkl>(box, rinput, cinput, rreference, creference);
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_1d_reorder_one<backend::cufft>(box, rinput, cinput, rreference, creference);
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_1d_reorder_one<backend::rocfft>(box, rinput, cinput, rreference, creference);
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_1d_reorder_one<backend::onemkl>(box, rinput, cinput, rreference, creference);
    #endif
}

// Tests the transpose reshape when applied within a single node (without MPI).
template<typename backend_tag>
void test_in_node_transpose(){
    using scalar_type = double;
    using vcontainer = typename test_traits<backend_tag>::template container<scalar_type>;
    using ltag = typename backend::buffer_traits<backend_tag>::location;
    current_test<scalar_type, using_nompi> name("reshape transpose");

    backend::device_instance<ltag> device;
    std::vector<int> proc, offset, sizes; // dummy variables, only needed to call the overlap map method
    std::vector<heffte::pack_plan_3d<int>> plans;

    box3d<> inbox(std::array<int, 3>{0, 0, 0}, std::array<int, 3>{1, 2, 3}, std::array<int, 3>{0, 1, 2}); // reference box
    auto const input = make_input<scalar_type>(); // reference input

    // test 1, transpose the data to order (1, 2, 0)
    box3d<> destination1(std::array<int, 3>{0, 0, 0}, std::array<int, 3>{1, 2, 3}, std::array<int, 3>{1, 2, 0});
    heffte::compute_overlap_map_transpose_pack(0, 1, destination1, {inbox}, proc, offset, sizes, plans);

    std::vector<scalar_type> reference = {1.0, 3.0, 5.0, 7.0,  9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
                                          2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0};

    auto active_intput = test_traits<backend_tag>::load(input);
    vcontainer result(24);

    // when doing out-of-place transpose we do not need a workspace and can use a nullptr instead,
    // but the in/out-place is checked at runtime  and one of the branches that is never used
    // will still be compiled with the hard-coded nullptr, to suppress the warning we are using a dummy-null dnull
    scalar_type *dnull = result.data();

    heffte::reshape3d_transpose<ltag, int>(device.stream(), plans[0]).apply(1, active_intput.data(), result.data(), dnull);

    sassert(match(result, reference));

    // test 2, transpose the data to order (2, 1, 0)
    box3d<> destination2(std::array<int, 3>{0, 0, 0}, std::array<int, 3>{1, 2, 3}, std::array<int, 3>{2, 1, 0});
    plans.clear();
    heffte::compute_overlap_map_transpose_pack(0, 1, destination2, {inbox}, proc, offset, sizes, plans);
    heffte::reshape3d_transpose<ltag, int>(device.stream(), plans[0]).apply(1, active_intput.data(), result.data(), dnull);

    reference = {1.0,  7.0, 13.0, 19.0,  3.0,  9.0, 15.0, 21.0,  5.0, 11.0, 17.0, 23.0,
                 2.0,  8.0, 14.0, 20.0,  4.0, 10.0, 16.0, 22.0,  6.0, 12.0, 18.0, 24.0};
    sassert(match(result, reference));

    // flip back the data
    plans.clear();
    heffte::compute_overlap_map_transpose_pack(0, 1, inbox, {destination2}, proc, offset, sizes, plans);
    auto active_reference = test_traits<backend_tag>::load(reference);
    heffte::reshape3d_transpose<ltag, int>(device.stream(), plans[0]).apply(1, active_reference.data(), result.data(), dnull);
    sassert(match(result, input));

    // test 3, transpose the data to order (0, 2, 1)
    box3d<> destination3(std::array<int, 3>{0, 0, 0}, std::array<int, 3>{1, 2, 3}, std::array<int, 3>{0, 2, 1});
    plans.clear();
    heffte::compute_overlap_map_transpose_pack(0, 1, destination3, {inbox}, proc, offset, sizes, plans);
    heffte::reshape3d_transpose<ltag, int>(device.stream(), plans[0]).apply(1, active_intput.data(), result.data(), dnull);

    reference = {1.0, 2.0,  7.0,  8.0, 13.0, 14.0, 19.0, 20.0,
                 3.0, 4.0,  9.0, 10.0, 15.0, 16.0, 21.0, 22.0,
                 5.0, 6.0, 11.0, 12.0, 17.0, 18.0, 23.0, 24.0};
    sassert(match(result, reference));
}
// Instantiates the in-node transpose test for all available backends.
void test_transpose(){
    test_in_node_transpose<backend::stock>();
    #ifdef Heffte_ENABLE_FFTW
    test_in_node_transpose<backend::fftw>();
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_in_node_transpose<backend::mkl>();
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_in_node_transpose<backend::cufft>();
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_in_node_transpose<backend::rocfft>();
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_in_node_transpose<backend::onemkl>();
    #endif
}

/*
 * Make data for a world box using a uniform random distribution on (0, 1).
 * The random seed is fixed, thus the result is deterministic and repeatable.
 */
template<typename scalar_type>
std::vector<scalar_type> make_data(box3d<> const world){
    std::minstd_rand park_miller(4242);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<scalar_type> result(world.count());
    for(auto &r : result)
        r = static_cast<scalar_type>(unif(park_miller));
    return result;
}

template<typename backend_a, typename backend_b, typename precision_type>
void test_cross_reference(){
    current_test<precision_type, using_nompi> name(backend::name<backend_a>() + " - " + backend::name<backend_b>() + " reference");

    using location_a = typename backend::buffer_traits<backend_a>::location;
    using location_b = typename backend::buffer_traits<backend_b>::location;

    backend::device_instance<location_a> device_a;
    backend::device_instance<location_b> device_b;

    box3d<> box = {{0, 0, 0}, {42, 75, 23}};

    auto rinput = make_data<precision_type>(box);
    auto cinput = make_data<std::complex<precision_type>>(box);

    auto rinput_a = test_traits<location_a>::load(rinput);
    auto rinput_b = test_traits<location_b>::load(rinput);

    auto cinput_a = test_traits<location_a>::load(cinput);
    auto cinput_b = test_traits<location_b>::load(cinput);

    // different backends can differ in the computed results, especially in single precision
    // the test in approx is very strict to disallow numerical error in the MPI communication routines
    // the test here has to be more permissive to allow for the variability in the different backends
    // the numbers below allow for 3.5 additional digits of difference for single precision and 2 for double precision
    double correction = (std::is_same<precision_type, float>::value) ? 0.0005 : 0.01;
    if (std::is_same<backend_a, backend::stock>::value or std::is_same<backend_b, backend::stock>::value)
        correction *= 0.1; // the stock backend is the most different,  allow for another digit of variability

    for(int i=0; i<3; i++){
        auto fft_a = make_executor<backend_a>(device_a.stream(), box, i);
        auto fft_b = make_executor<backend_b>(device_b.stream(), box, i);

        auto workspace_a = make_buffer_container<std::complex<precision_type>>(device_a.stream(), fft_a->workspace_size());
        auto workspace_b = make_buffer_container<std::complex<precision_type>>(device_b.stream(), fft_b->workspace_size());

        fft_a->forward(cinput_a.data(), workspace_a.data());
        fft_b->forward(cinput_b.data(), workspace_b.data());
        sassert(approx(cinput_a, cinput_b, correction));

        fft_a->backward(cinput_a.data(), workspace_a.data());
        fft_b->backward(cinput_b.data(), workspace_b.data());
        sassert(approx(cinput_a, cinput_b, correction));

        fft_a->forward(rinput_a.data(), cinput_a.data(), workspace_a.data());
        fft_b->forward(rinput_b.data(), cinput_b.data(), workspace_b.data());
        sassert(approx(cinput_a, cinput_b, correction));

        fft_a->backward(cinput_a.data(), rinput_a.data(), workspace_a.data());
        fft_b->backward(cinput_b.data(), rinput_b.data(), workspace_b.data());
        sassert(approx(rinput_a, rinput_b, correction));
    }
}
template<typename backend_a, typename backend_b, typename precision_type>
void test_cross_reference_r2c(){
    current_test<precision_type, using_nompi> name(backend::name<backend_a>() + " - " + backend::name<backend_b>() + " reference r2c");

    using location_a = typename backend::buffer_traits<backend_a>::location;
    using location_b = typename backend::buffer_traits<backend_b>::location;

    backend::device_instance<location_a> device_a;
    backend::device_instance<location_b> device_b;

    // see the c2c variant above
    double correction = (std::is_same<precision_type, float>::value) ? 0.0005 : 0.01;
    if (std::is_same<backend_a, backend::stock>::value or std::is_same<backend_b, backend::stock>::value)
        correction *= 0.1;

    for(int case_counter = 0; case_counter < 2; case_counter++){
        // due to alignment issues on some backends (cufft)
        // need to check the case when both size[0] and size[1] are odd
        //                        when at least one is even
        box3d<> box = (case_counter == 0) ?
                       box3d<>({0, 0, 0}, {42, 70, 21}) :
                       box3d<>({0, 0, 0}, {41, 50, 21});

        auto rinput = make_data<precision_type>(box);

        auto rinput_a = test_traits<location_a>::load(rinput);
        auto rinput_b = test_traits<location_b>::load(rinput);

        for(int i=0; i<3; i++){
            if (std::is_same<backend_a, backend::rocfft>::value or std::is_same<backend_b, backend::rocfft>::value)
                continue; // rocfft r2c work only in direction 0

            auto fft_a = make_executor_r2c<backend_a>(device_a.stream(), box, i);
            auto fft_b = make_executor_r2c<backend_b>(device_b.stream(), box, i);

            auto result_a = make_buffer_container<std::complex<precision_type>>(device_a.stream(), box.r2c(i).count());
            auto result_b = make_buffer_container<std::complex<precision_type>>(device_b.stream(), box.r2c(i).count());

            auto workspace_a = make_buffer_container<std::complex<precision_type>>(device_a.stream(), fft_a->workspace_size());
            auto workspace_b = make_buffer_container<std::complex<precision_type>>(device_b.stream(), fft_b->workspace_size());

            fft_a->forward(rinput_a.data(), result_a.data(), workspace_a.data());
            fft_b->forward(rinput_b.data(), result_b.data(), workspace_b.data());
            sassert(approx(result_a, result_b, correction));

            fft_a->backward(result_a.data(), rinput_a.data(), workspace_a.data());
            fft_b->backward(result_b.data(), rinput_b.data(), workspace_b.data());
            sassert(approx(rinput_a, rinput_b, correction));
        }
    }
}

void test_cross_reference(){
    #ifdef Heffte_ENABLE_FFTW
    test_cross_reference<backend::stock, backend::fftw, float>();
    test_cross_reference<backend::stock, backend::fftw, double>();
    test_cross_reference_r2c<backend::stock, backend::fftw, float>();
    test_cross_reference_r2c<backend::stock, backend::fftw, double>();
    #endif
    #if defined(Heffte_ENABLE_FFTW) and defined(Heffte_ENABLE_MKL)
    test_cross_reference<backend::fftw, backend::mkl, float>();
    test_cross_reference<backend::fftw, backend::mkl, double>();
    test_cross_reference_r2c<backend::fftw, backend::mkl, float>();
    test_cross_reference_r2c<backend::fftw, backend::mkl, double>();
    #endif
    #if defined(Heffte_ENABLE_FFTW) and defined(Heffte_ENABLE_CUDA)
    test_cross_reference<backend::fftw, backend::cufft, float>();
    test_cross_reference<backend::fftw, backend::cufft, double>();
    test_cross_reference_r2c<backend::fftw, backend::cufft, float>();
    test_cross_reference_r2c<backend::fftw, backend::cufft, double>();
    #endif
    #if defined(Heffte_ENABLE_FFTW) and defined(Heffte_ENABLE_ROCM)
    test_cross_reference<backend::fftw, backend::rocfft, float>();
    test_cross_reference<backend::fftw, backend::rocfft, double>();
    test_cross_reference_r2c<backend::fftw, backend::rocfft, float>();
    test_cross_reference_r2c<backend::fftw, backend::rocfft, double>();
    #endif
    #if defined(Heffte_ENABLE_MKL) and defined(Heffte_ENABLE_ONEAPI)
    test_cross_reference<backend::mkl, backend::onemkl, float>();
    test_cross_reference<backend::mkl, backend::onemkl, double>();
    test_cross_reference_r2c<backend::mkl, backend::onemkl, float>();
    test_cross_reference_r2c<backend::mkl, backend::onemkl, double>();
    #endif
}

int main(int, char**){

    all_tests<using_nompi> name("Non-MPI Tests");

    test_factorize();
    test_process_grid();
    test_split_pencils();
    test_cpu_scale();

    test_gpu_vector();
    test_gpu_scale();

    test_1d_all();

    test_1d_reorder();
    test_transpose();

    test_cross_reference();

    return 0;
}
