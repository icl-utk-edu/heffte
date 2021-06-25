/**
 * @class
 * heFFTe kernels for complex-to-complex transforms
 */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_fft3d.h"

#define heffte_instantiate_fft3d(some_backend, index) \
    template class fft3d<some_backend, index>; \
    template void fft3d<some_backend, index>::standard_transform<float>(float const[], std::complex<float>[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend, index>::standard_transform<double>(double const[], std::complex<double>[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \
    template void fft3d<some_backend, index>::standard_transform<float>(std::complex<float> const[], float[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend, index>::standard_transform<double>(std::complex<double> const[], double[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \
    template void fft3d<some_backend, index>::standard_transform<float>(std::complex<float> const[], std::complex<float>[], std::complex<float>[], \
                                                                 std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                 direction, scale  \
                                                                ) const;    \
    template void fft3d<some_backend, index>::standard_transform<double>(std::complex<double> const[], std::complex<double>[], std::complex<double>[], \
                                                                  std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &, std::array<backend_executor*, 3> const, \
                                                                  direction, scale \
                                                                 ) const;   \

namespace heffte {

template<typename backend_tag, typename index>
fft3d<backend_tag, index>::fft3d(logic_plan3d<index> const &plan, int const this_mpi_rank, MPI_Comm const comm) :
    backend::device_instance<backend_tag>(),
    pinbox(new box3d<index>(plan.in_shape[0][this_mpi_rank])), poutbox(new box3d<index>(plan.out_shape[3][this_mpi_rank])),
    scale_factor(1.0 / static_cast<double>(plan.index_count))
    #ifdef Heffte_ENABLE_MAGMA
    , hmagma(this->stream())
    #endif
{
    setup(plan, comm);
}

template<typename backend_tag, typename index>
fft3d<backend_tag, index>::fft3d(typename backend::device_instance<backend_tag>::stream_type gpu_stream,
                                logic_plan3d<index> const &plan, int const this_mpi_rank, MPI_Comm const comm) :
    backend::device_instance<backend_tag>(gpu_stream),
    pinbox(new box3d<index>(plan.in_shape[0][this_mpi_rank])), poutbox(new box3d<index>(plan.out_shape[3][this_mpi_rank])),
    scale_factor(1.0 / static_cast<double>(plan.index_count))
    #ifdef Heffte_ENABLE_MAGMA
    , hmagma(this->stream())
    #endif
{
    setup(plan, comm);
}

template<typename backend_tag, typename index>
void fft3d<backend_tag, index>::setup(logic_plan3d<index> const &plan, MPI_Comm const comm){
    for(int i=0; i<4; i++){
        forward_shaper[i]    = make_reshape3d<backend_tag>(this->stream(), plan.in_shape[i], plan.out_shape[i], comm, plan.options);
        backward_shaper[3-i] = make_reshape3d<backend_tag>(this->stream(), plan.out_shape[i], plan.in_shape[i], comm, plan.options);
    }

    fft0 = one_dim_backend<backend_tag>::make(this->stream(), plan.out_shape[0][mpi::comm_rank(comm)], plan.fft_direction[0]);
    fft1 = one_dim_backend<backend_tag>::make(this->stream(), plan.out_shape[1][mpi::comm_rank(comm)], plan.fft_direction[1]);
    fft2 = one_dim_backend<backend_tag>::make(this->stream(), plan.out_shape[2][mpi::comm_rank(comm)], plan.fft_direction[2]);
}

template<typename backend_tag, typename index>
template<typename scalar_type> // complex to complex case
void fft3d<backend_tag, index>::standard_transform(std::complex<scalar_type> const input[], std::complex<scalar_type> output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                                            std::array<backend_executor*, 3> const executor,
                                            direction dir, scale scaling) const{
    /*
     * The logic is a bit messy, but the objective is:
     * - call all shaper and executor objects in the correct order
     * - assume that any or all of the shapers can be missing, i.e., null unique_ptr()
     * - do not allocate buffers if not needed
     * - never have more than 2 allocated buffers (input and output)
     */
    auto apply_fft = [&](int i, std::complex<scalar_type> data[])
        ->void{
            add_trace name("fft-1d");
            if (dir == direction::forward){
                executor[i]->forward(data);
            }else{
                executor[i]->backward(data);
            }
        };

    int num_active = count_active(shaper);
    int last = get_last_active(shaper);

    if (last < 1){ // no extra buffer case
        add_trace name("reshape/copy");
        // move input -> output and apply all ffts
        // use either zeroth shaper or simple copy (or nothing in case of in-place transform)
        if (last == 0){
            shaper[0]->apply(input, output, workspace);
        }else if (input != output){
            backend::data_manipulator<location_tag>::copy_n(this->stream(), input, executor[0]->box_size(), output);
        }
        for(int i=0; i<3; i++)
            apply_fft(i, output);

        apply_scale(dir, scaling, output);
        return;
    }

    // with only one reshape, the temp buffer would be used only if not doing in-place
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();
    if (num_active == 1){ // one active and not shaper 0
        std::complex<scalar_type> *effective_input = output;
        if (input != output){
            add_trace name("copy");
            backend::data_manipulator<location_tag>::copy_n(this->stream(), input, executor[0]->box_size(), temp_buffer);
            effective_input = temp_buffer;
        }
        for(int i=0; i<last; i++)
            apply_fft(i, effective_input);
        { add_trace name("reshape");
        shaper[last]->apply(effective_input, output, workspace);
        }
        for(int i=last; i<3; i++)
            apply_fft(i, output);

        apply_scale(dir, scaling, output);
        return;
    }

    // with two or more reshapes, the first reshape must move to the temp_buffer and the last must move to output
    int active_shaper = 0;
    if (shaper[0] or input != output){
        if (shaper[0]){
            add_trace name("reshape");
            shaper[0]->apply(input, temp_buffer, workspace);
        }else{
            add_trace name("copy");
            backend::data_manipulator<location_tag>::copy_n(this->stream(), input, executor[0]->box_size(), temp_buffer);
        }
        active_shaper = 1;
    }else{
        // in place transform and shaper[0] is not active
        while(not shaper[active_shaper]){
            // note, at least one shaper must be active, otherwise last will catch it
            apply_fft(active_shaper++, output);
        }
        { add_trace name("reshape");
        shaper[active_shaper]->apply(output, temp_buffer, workspace);
        }
        active_shaper += 1;
    }
    apply_fft(active_shaper - 1, temp_buffer); // one reshape was applied above

    for(int i=active_shaper; i<last; i++){
        if (shaper[i]){
            add_trace name("reshape");
            shaper[i]->apply(temp_buffer, temp_buffer, workspace);
        }
        apply_fft(i, temp_buffer);
    }
    { add_trace name("reshape");
    shaper[last]->apply(temp_buffer, output, workspace);
    }

    for(int i=last; i<3; i++)
        apply_fft(i, output);

    apply_scale(dir, scaling, output);
}
template<typename backend_tag, typename index>
template<typename scalar_type> // real to complex case
void fft3d<backend_tag, index>::standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                                            std::array<typename one_dim_backend<backend_tag>::type*, 3> const executor,
                                            direction, scale scaling) const{
    /*
     * Follows logic similar to the complex-to-complex case but the first shaper and executor will be applied to real data.
     * This is the real-to-complex variant which is possible only for a forward transform,
     * thus the direction parameter is ignored.
     */
    int last = get_last_active(shaper);

    scalar_type* reshaped_input = reinterpret_cast<scalar_type*>(workspace);
    scalar_type const *effective_input = input; // either input or the result of reshape operation 0
    if (shaper[0]){
        add_trace name("reshape");
        shaper[0]->apply(input, reshaped_input, reinterpret_cast<scalar_type*>(workspace + get_max_size(executor)));
        effective_input = reshaped_input;
    }

    if (last < 1){ // no reshapes after 0
        add_trace name("fft-1d x3");
        executor[0]->forward(effective_input, output);
        executor[1]->forward(output);
        executor[2]->forward(output);
        apply_scale(direction::forward, scaling, output);
        return;
    }

    // if there is messier combination of transforms, then we need internal buffers
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();
    { add_trace name("fft-1d");
    executor[0]->forward(effective_input, temp_buffer);
    }

    for(int i=1; i<last; i++){
        if (shaper[i]){
            add_trace name("reshape");
            shaper[i]->apply(temp_buffer, temp_buffer, workspace);
        }
        add_trace name("fft-1d");
        executor[i]->forward(temp_buffer);
    }
    { add_trace name("reshape");
    shaper[last]->apply(temp_buffer, output, workspace);
    }

    for(int i=last; i<3; i++){
        add_trace name("fft-1d");
        executor[i]->forward(output);
    }

    apply_scale(direction::forward, scaling, output);
}
template<typename backend_tag, typename index>
template<typename scalar_type> // complex to real case
void fft3d<backend_tag, index>::standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                                            std::complex<scalar_type> workspace[],
                                            std::array<std::unique_ptr<reshape3d_base<index>>, 4> const &shaper,
                                            std::array<backend_executor*, 3> const executor, direction, scale scaling) const{
    /*
     * Follows logic similar to the complex-to-complex case but the last shaper and executor will be applied to real data.
     * This is the complex-to-real variant which is possible only for a backward transform,
     * thus the direction parameter is ignored.
     */
    std::complex<scalar_type> *temp_buffer = workspace + size_comm_buffers();

    if (shaper[0]){
        add_trace name("reshape");
        shaper[0]->apply(input, temp_buffer, workspace);
    }else{
        add_trace name("copy");
        backend::data_manipulator<location_tag>::copy_n(this->stream(), input, executor[0]->box_size(), temp_buffer);
    }

    for(int i=0; i<2; i++){ // apply the two complex-to-complex ffts
        { add_trace name("fft-1d x3");
        executor[i]->backward(temp_buffer);
        }
        if (shaper[i+1]){
            add_trace name("reshape");
            shaper[i+1]->apply(temp_buffer, temp_buffer, workspace);
        }
    }

    // the result of the first two ffts and three reshapes is stored in temp_buffer
    // executor 2 must apply complex to real backward transform
    if (shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        scalar_type* real_buffer = reinterpret_cast<scalar_type*>(temp_buffer + executor[2]->box_size());
        { add_trace name("fft-1d");
        executor[2]->backward(temp_buffer, real_buffer);
        }
        add_trace name("reshape");
        shaper[3]->apply(real_buffer, output, reinterpret_cast<scalar_type*>(workspace));
    }else{
        add_trace name("fft-1d");
        executor[2]->backward(temp_buffer, output);
    }

    apply_scale(direction::backward, scaling, output);
}

heffte_instantiate_fft3d(backend::stock, int)
heffte_instantiate_fft3d(backend::stock, long long)
#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_fft3d(backend::fftw, int)
heffte_instantiate_fft3d(backend::fftw, long long)
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_fft3d(backend::mkl, int)
heffte_instantiate_fft3d(backend::mkl, long long)
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_fft3d(backend::cufft, int)
heffte_instantiate_fft3d(backend::cufft, long long)
#endif
#ifdef Heffte_ENABLE_ROCM
heffte_instantiate_fft3d(backend::rocfft, int)
heffte_instantiate_fft3d(backend::rocfft, long long)
#endif
#ifdef Heffte_ENABLE_ONEAPI
heffte_instantiate_fft3d(backend::onemkl, int)
heffte_instantiate_fft3d(backend::onemkl, long long)
#endif

}
