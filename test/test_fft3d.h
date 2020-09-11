/** @class */
/*
    -- heFFTe --
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

template<typename scalar_type>
std::vector<scalar_type> rescale(box3d const world, std::vector<scalar_type> const &data, scale scaling){
    std::vector<scalar_type> result = data;
    double scaling_factor = (scaling == scale::none) ? 1.0 : 1.0 / static_cast<double>(world.count());
    if (scaling == scale::symmetric) scaling_factor = std::sqrt(scaling_factor);
    if (scaling != scale::none)
        data_scaling<tag::cpu>::apply(result.size(), result.data(), scaling_factor);
    return result;
}

template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> convert_to_output(std::vector<scalar_type> const &data){
    std::vector<typename fft_output<scalar_type>::type> result(data.size());
    for(size_t i=0; i<data.size(); i++) result[i] = static_cast<typename fft_output<scalar_type>::type>(data[i]);
    return result;
}

template<typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> get_complex_subbox(box3d const world, box3d const box, std::vector<scalar_type> const &input){
    return get_subbox(world, box, convert_to_output(input));
}

#ifdef Heffte_ENABLE_FFTW
template<typename scalar_type>
std::vector<scalar_type> compute_fft_fftw(box3d const world, std::vector<scalar_type> const &input){
    assert(input.size() == static_cast<size_t>(world.count()));
    std::vector<scalar_type> result = input;
    for(int i=0; i<3; i++)
        heffte::fftw_executor(world, i).forward(result.data());
    return result;
}
#endif

#ifdef Heffte_ENABLE_MKL
template<typename scalar_type>
std::vector<scalar_type> compute_fft_mkl(box3d const world, std::vector<scalar_type> const &input){
    assert(input.size() == world.count());
    std::vector<scalar_type> result = input;
    for(int i=0; i<3; i++)
        heffte::mkl_executor(world, i).forward(result.data());
    return result;
}
#endif

template<typename backend_tag, typename scalar_type>
struct perform_fft{};
template<typename backend_tag, typename scalar_type>
struct input_maker{};

#ifdef Heffte_ENABLE_FFTW
template<typename scalar_type>
struct perform_fft<backend::fftw, scalar_type>{
    static std::vector<typename fft_output<scalar_type>::type> forward(box3d const world, std::vector<scalar_type> const &input){
        std::vector<typename fft_output<scalar_type>::type> result(input.size());
        for(size_t i=0; i<input.size(); i++)
            result[i] = static_cast<typename fft_output<scalar_type>::type>(input[i]);
        return compute_fft_fftw(world, result);
    }
    static std::vector<std::complex<scalar_type>> forward(box3d const world, std::vector<std::complex<scalar_type>> const &input){
        return compute_fft_fftw(world, input);
    }
};
template<typename scalar_type>
struct input_maker<backend::fftw, scalar_type>{
    static std::vector<scalar_type> select(box3d const world, box3d const box, std::vector<scalar_type> const &input){
        return get_subbox(world, box, input);
    }
};
#endif

#ifdef Heffte_ENABLE_MKL
template<typename scalar_type>
struct perform_fft<backend::mkl, scalar_type>{
    static std::vector<typename fft_output<scalar_type>::type> forward(box3d const world, std::vector<scalar_type> const &input){
        std::vector<typename fft_output<scalar_type>::type> result(input.size());
        for(size_t i=0; i<input.size(); i++)
            result[i] = static_cast<typename fft_output<scalar_type>::type>(input[i]);
        return compute_fft_mkl(world, result);
    }
    static std::vector<std::complex<scalar_type>> forward(box3d const world, std::vector<std::complex<scalar_type>> const &input){
        return compute_fft_mkl(world, input);
    }
};
template<typename scalar_type>
struct input_maker<backend::mkl, scalar_type>{
    static std::vector<scalar_type> select(box3d const world, box3d const box, std::vector<scalar_type> const &input){
        return get_subbox(world, box, input);
    }
};
#endif


#ifdef Heffte_ENABLE_CUDA
template<typename scalar_type>
cuda::vector<scalar_type> compute_fft_cufft(box3d const world, cuda::vector<scalar_type> const &input){
    assert(input.size() == static_cast<size_t>(world.count()));
    cuda::vector<scalar_type> result = input;
    for(int i=0; i<3; i++)
        heffte::cufft_executor(world, i).forward(result.data());
    return result;
}
template<typename scalar_type>
cuda::vector<typename fft_output<scalar_type>::type> compute_fft_cuda(box3d const world, cuda::vector<scalar_type> const &input){
    cuda::vector<typename fft_output<scalar_type>::type> result(input.size());
    cuda::convert(input.size(), input.data(), result.data());
    return compute_fft_cufft(world, result);
}
template<typename scalar_type>
cuda::vector<std::complex<scalar_type>> compute_fft_cuda(box3d const world, cuda::vector<std::complex<scalar_type>> const &input){
    return compute_fft_cufft(world, input);
}
template<typename scalar_type>
struct perform_fft<backend::cufft, scalar_type>{
    static std::vector<typename fft_output<scalar_type>::type> forward(box3d const world, std::vector<scalar_type> const &input){
        return cuda::unload(compute_fft_cuda(world, cuda::load(input)));
    }
};
template<typename scalar_type>
struct input_maker<backend::cufft, scalar_type>{
    static cuda::vector<scalar_type> select(box3d const world, box3d const box, std::vector<scalar_type> const &input){
        return cuda::load(get_subbox(world, box, input));
    }
};
template<typename scalar_type>
std::vector<scalar_type> rescale(box3d const world, cuda::vector<scalar_type> const &data, scale scaling){
    return rescale(world, cuda::unload(data), scaling);
}
#endif

#ifdef Heffte_ENABLE_ROCM
template<typename scalar_type>
rocm::vector<scalar_type> compute_fft_rocfft(box3d const world, rocm::vector<scalar_type> const &input){
    assert(input.size() == static_cast<size_t>(world.count()));
    rocm::vector<scalar_type> result = input;
    for(int i=0; i<3; i++)
        heffte::rocfft_executor(world, i).forward(result.data());
    return result;
}
template<typename scalar_type>
rocm::vector<typename fft_output<scalar_type>::type> compute_fft_rocm(box3d const world, rocm::vector<scalar_type> const &input){
    rocm::vector<typename fft_output<scalar_type>::type> result(input.size());
    rocm::convert(input.size(), input.data(), result.data());
    return compute_fft_rocfft(world, result);
}
template<typename scalar_type>
rocm::vector<std::complex<scalar_type>> compute_fft_rocm(box3d const world, rocm::vector<std::complex<scalar_type>> const &input){
    return compute_fft_rocfft(world, input);
}
template<typename scalar_type>
struct perform_fft<backend::rocfft, scalar_type>{
    static std::vector<typename fft_output<scalar_type>::type> forward(box3d const world, std::vector<scalar_type> const &input){
        return rocm::unload(compute_fft_rocm(world, rocm::load(input)));
    }
};
template<typename scalar_type>
struct input_maker<backend::rocfft, scalar_type>{
    static rocm::vector<scalar_type> select(box3d const world, box3d const box, std::vector<scalar_type> const &input){
        return rocm::load(get_subbox(world, box, input));
    }
};
template<typename scalar_type>
std::vector<scalar_type> rescale(box3d const world, rocm::vector<scalar_type> const &data, scale scaling){
    return rescale(world, rocm::unload(data), scaling);
}
#endif

template<typename backend_tag, typename scalar_type>
std::vector<typename fft_output<scalar_type>::type> forward_fft(box3d const world, std::vector<scalar_type> const &input){
    return perform_fft<backend_tag, scalar_type>::forward(world, input);
}

template<typename backend_tag>
void test_fft3d_const_dest2(MPI_Comm comm){
    assert(mpi::comm_size(comm) == 2);
    current_test<int, using_mpi, backend_tag> name("constructor heffte::fft3d", comm);
    box3d const world = {{0, 0, 0}, {4, 4, 4}};
    std::vector<box3d> boxes = heffte::split_world(world, {2, 1, 1});
    int const me = mpi::comm_rank(comm);
    // construct an instance of heffte::fft3d and delete it immediately
    heffte::fft3d<backend_tag> fft(boxes[me], boxes[me], comm);
}

template<typename backend_tag, typename scalar_type, int h0, int h1, int h2>
void test_fft3d_vectors(MPI_Comm comm){
    // works with ranks 6 and 8 only
    int const num_ranks = mpi::comm_size(comm);
    assert(num_ranks == 6 or num_ranks == 8);
    current_test<scalar_type, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test heffte::fft3d", comm);
    int const me = mpi::comm_rank(comm);
    box3d const world = {{0, 0, 0}, {h0, h1, h2}};
    auto world_input = make_data<scalar_type>(world);
    auto world_complex = convert_to_output(world_input);
    std::vector<decltype(std::real(world_complex[0]))> world_real(world_input.size());
    for(size_t i=0; i<world_real.size(); i++) world_real[i] = std::real(world_input[i]);
    auto world_fft = forward_fft<backend_tag>(world, world_input);

    std::array<heffte::scale, 3> fscale = {heffte::scale::none, heffte::scale::symmetric, heffte::scale::full};
    std::array<heffte::scale, 3> bscale = {heffte::scale::full, heffte::scale::symmetric, heffte::scale::none};

    for(auto const &options : make_all_options<backend_tag>()){
    for(int i=0; i<3; i++){
        std::array<int, 3> split = {1, 1, 1};
        if (num_ranks == 6){
            split[i] = 2;
            split[(i+1) % 3] = 3;
        }else if (num_ranks == 8){
            split = {2, 2, 2};
        }
        std::vector<box3d> boxes = heffte::split_world(world, split);
        assert(boxes.size() == static_cast<size_t>(num_ranks));

        // get a semi-random inbox and outbox
        // makes sure that the boxes do not have to match
        int iindex = me, oindex = me; // indexes of the input and outboxes
        if (num_ranks == 6){ // shuffle the boxes
            iindex = (me+2) % num_ranks;
            oindex = (me+3) % num_ranks;
        }else if (num_ranks == 8){
            iindex = (me+3) % num_ranks;
            oindex = (me+5) % num_ranks;
        }

        box3d const inbox  = boxes[iindex];
        box3d const outbox = boxes[oindex];

        auto local_input         = input_maker<backend_tag, scalar_type>::select(world, inbox, world_input);
        auto local_complex_input = get_subbox(world, inbox, world_complex);
        auto local_real_input    = get_subbox(world, inbox, world_real);
        auto reference_fft       = rescale(world, get_subbox(world, outbox, world_fft), fscale[i]);

        heffte::fft3d<backend_tag> fft(inbox, outbox, comm, options);

        auto result = fft.forward(local_input, fscale[i]);
        tassert(approx(result, reference_fft));

        auto backward_cresult = fft.backward(result, bscale[i]);
        auto backward_scaled_cresult = rescale(world, backward_cresult, scale::none);
        tassert(approx(local_complex_input, backward_scaled_cresult));

        auto backward_rresult = fft.backward_real(result, bscale[i]);
        auto backward_scaled_rresult = rescale(world, backward_rresult, scale::none);
        tassert(approx(backward_scaled_rresult, local_real_input));
    }
    } // different option variants
}

template<typename backend_tag, typename scalar_type, int h0, int h1, int h2>
void test_fft3d_arrays(MPI_Comm comm){
    using output_type = typename fft_output<scalar_type>::type; // complex type of the output
    using input_container  = typename heffte::fft3d<backend_tag>::template buffer_container<scalar_type>; // std::vector or cuda::vector
    using output_container = typename heffte::fft3d<backend_tag>::template buffer_container<output_type>; // std::vector or cuda::vector

    // works with ranks 2 and 12 only
    int const num_ranks = mpi::comm_size(comm);
    assert(num_ranks == 1 or num_ranks == 2 or num_ranks == 12);
    current_test<scalar_type, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test heffte::fft3d", comm);

    int const me = mpi::comm_rank(comm);
    box3d const world = {{0, 0, 0}, {h0, h1, h2}};
    auto world_input   = make_data<scalar_type>(world);
    auto world_complex = convert_to_output(world_input); // if using real input, convert to the complex output type
    auto world_fft     = forward_fft<backend_tag>(world, world_input); // compute reference fft

    for(auto const &options : make_all_options<backend_tag>()){
    for(int i=0; i<3; i++){
        // split the world into processors
        std::array<int, 3> split = {1, 1, 1};
        if (num_ranks == 2){
            split[i] = 2;
        }else if (num_ranks == 12){
            split = {2, 2, 2};
            split[i] = 3;
        }
        std::vector<box3d> boxes = heffte::split_world(world, split);
        assert(boxes.size() == static_cast<size_t>(num_ranks));

        // get the local input as a cuda::vector or std::vector
        auto local_input = input_maker<backend_tag, scalar_type>::select(world, boxes[me], world_input);
        auto reference_fft = get_subbox(world, boxes[me], world_fft); // reference solution
        output_container forward(local_input.size()); // computed solution

        heffte::fft3d<backend_tag> fft(boxes[me], boxes[me], comm, options);

        output_container workspace(fft.size_workspace());

        fft.forward(local_input.data(), forward.data(), workspace.data()); // compute the forward fft
        tassert(approx(forward, reference_fft)); // compare to the reference

        input_container rbackward(local_input.size()); // compute backward fft using scalar_type
        fft.backward(forward.data(), rbackward.data(), workspace.data());
        auto backward_result = rescale(world, rbackward, scale::full); // always std::vector
        tassert(approx(local_input, backward_result)); // compare with the original input

        output_container cbackward(local_input.size()); // complex backward transform
        fft.backward(forward.data(), cbackward.data(), workspace.data());
        auto cbackward_result = rescale(world, cbackward, scale::full);
        // convert the world to complex numbers and extract the reference sub-box
        tassert(approx(get_complex_subbox(world, boxes[me], world_input),
                       cbackward_result));

        output_container inplace_buffer(std::max(fft.size_inbox(), fft.size_outbox()));
        data_manipulator<typename fft3d<backend_tag>::location_tag>::copy_n(local_input.data(), fft.size_inbox(), inplace_buffer.data());
        fft.forward(inplace_buffer.data(), inplace_buffer.data());
        output_container inplace_forward(inplace_buffer.data(), inplace_buffer.data() + fft.size_outbox());
        tassert(approx(inplace_forward, reference_fft)); // compare to the reference

        auto inplace_buffer_copy = inplace_buffer;
        fft.backward(inplace_buffer.data(), reinterpret_cast<scalar_type*>(inplace_buffer.data())); // in-place complex-to-real
        rbackward = input_container(reinterpret_cast<scalar_type*>(inplace_buffer.data()),
                                    reinterpret_cast<scalar_type*>(inplace_buffer.data()) + fft.size_inbox());
        backward_result = rescale(world, rbackward, scale::full); // always std::vector
        tassert(approx(local_input, backward_result));

        fft.backward(inplace_buffer_copy.data(), inplace_buffer_copy.data());
        output_container inplace_backward(inplace_buffer_copy.data(), inplace_buffer_copy.data() + fft.size_inbox());
        cbackward_result = rescale(world, inplace_backward, scale::full);
        tassert(approx(get_complex_subbox(world, boxes[me], world_input), cbackward_result));
    }
    } // different option variants
}


#endif
