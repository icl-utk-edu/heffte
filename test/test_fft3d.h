/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/
#ifndef HEFFTE_TEST_FFT
#define HEFFTE_TEST_FFT

#include <random>

#include "test_common.h"

template<typename scalar_type>
std::vector<scalar_type> make_data(box3d const world){
    std::minstd_rand park_miller(4242);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    std::vector<scalar_type> result(world.count());
    for(auto &r : result)
        r = static_cast<scalar_type>(unif(park_miller));
    return result;
}

template<typename scalar_type>
std::vector<scalar_type> get_subbox(box3d const world, box3d const box, std::vector<scalar_type> const &input){
    std::vector<scalar_type> result(box.count());
    int const mid_world = world.size[0];
    int const slow_world = world.size[0] * world.size[1];
    int const mid_box = box.size[0];
    int const slow_box = box.size[0] * box.size[1];
    for(int k=box.low[2]; k<=box.high[2]; k++){
        for(int j=box.low[1]; j<=box.high[1]; j++){
            for(int i=box.low[0]; i<=box.high[0]; i++){
                result[(k - box.low[2]) * slow_box + (j - box.low[1]) * mid_box + (i - box.low[0])]
                    = input[k * slow_world + j * mid_world + i];
            }
        }
    }
    return result;
}

#ifdef Heffte_ENABLE_FFTW
template<typename scalar_type>
std::vector<scalar_type> compute_fft_fftw(box3d const world, std::vector<scalar_type> const &input){
    assert(input.size() == world.count());
    std::vector<scalar_type> result = input;
    for(int i=0; i<3; i++)
        heffte::fftw_executor(world, i).forward(result.data());
    return result;
}
#endif

template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> compute_fft(box3d const world, std::vector<scalar_type> const &input){
    #ifdef Heffte_ENABLE_FFTW
    std::vector<typename fft_output<scalar_type>::type> result(input.size());
    for(size_t i=0; i<input.size(); i++)
        result[i] = static_cast<typename fft_output<scalar_type>::type>(input[i]);
    return compute_fft_fftw(world, result);
    #endif
}
template<typename scalar_type>
std::vector<std::complex<scalar_type>> compute_fft(box3d const world, std::vector<std::complex<scalar_type>> const &input){
    #ifdef Heffte_ENABLE_FFTW
    return compute_fft_fftw(world, input);
    #endif
}

template<typename backend_tag>
void test_fft3d_const_dest2(MPI_Comm comm){
    assert(mpi::comm_size(comm) == 2);
    current_test<> name("heffte::fft3d constructor", comm);
    box3d const world = {{0, 0, 0}, {4, 4, 4}};
    std::vector<box3d> boxes = heffte::split_world(world, {2, 1, 1});
    int const me = mpi::comm_rank(comm);
    // construct an instance of heffte::fft3d and delete it immediately
    heffte::fft3d<backend_tag> fft(boxes[me], boxes[me], comm);
}

template<typename backend_tag, typename scalar_type>
void test_fft3d_rank2(MPI_Comm comm){
    assert(mpi::comm_size(comm) == 2);
    current_test<scalar_type, using_mpi, backend_tag> name("-np 2 test heffte::fft3d", comm);
    using output_type = typename fft_output<scalar_type>::type;
    int const me = mpi::comm_rank(comm);
    box3d const world = {{0, 0, 0}, {9, 9, 9}};
    auto world_input = make_data<scalar_type>(world);
    auto world_fft = compute_fft(world, world_input);

    for(int i=0; i<3; i++){
        std::array<int, 3> split = {1, 1, 1};
        split[i] = 2;
        std::vector<box3d> boxes = heffte::split_world(world, split);
        assert(boxes.size() == 2);
        auto local_input = get_subbox(world, boxes[me], world_input);
        auto reference_fft = get_subbox(world, boxes[me], world_fft);
        std::vector<output_type> result(local_input.size());

        heffte::fft3d<backend_tag> fft(boxes[me], comm);

        fft.forward(local_input.data(), result.data());
        tassert(approx(result, reference_fft));

        std::vector<scalar_type> backward_result(local_input.size());
        fft.backward(result.data(), backward_result.data());
        for(auto &r : backward_result) r /= static_cast<scalar_type>(world.count());

        tassert(approx(backward_result, local_input));
    }
}


#endif
