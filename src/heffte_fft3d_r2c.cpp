/**
 * @class
 * heFFTe kernels for real-to-complex transforms
 */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/


#include "heffte_fft3d_r2c.h"

#define heffte_instantiate_fft3d_r2c(some_backend, index) \
    template class fft3d_r2c<some_backend, index>; \
    template void fft3d_r2c<some_backend, index>::standard_transform<float>(float const[], std::complex<float>[], std::complex<float>[], scale) const;    \
    template void fft3d_r2c<some_backend, index>::standard_transform<double>(double const[], std::complex<double>[], std::complex<double>[], scale) const; \
    template void fft3d_r2c<some_backend, index>::standard_transform<float>(std::complex<float> const[], float[], std::complex<float>[], scale) const;    \
    template void fft3d_r2c<some_backend, index>::standard_transform<double>(std::complex<double> const[], double[], std::complex<double>[], scale) const; \

namespace heffte {

template<typename backend_tag, typename index>
fft3d_r2c<backend_tag, index>::fft3d_r2c(logic_plan3d<index> const &plan, int const this_mpi_rank, MPI_Comm const comm) :
    pinbox(new box3d<index>(plan.in_shape[0][this_mpi_rank])), poutbox(new box3d<index>(plan.out_shape[3][this_mpi_rank])),
    scale_factor(1.0 / static_cast<double>(plan.index_count))
{
    for(int i=0; i<4; i++){
        forward_shaper[i]    = make_reshape3d<backend_tag>(plan.in_shape[i], plan.out_shape[i], comm, plan.options);
        backward_shaper[3-i] = make_reshape3d<backend_tag>(plan.out_shape[i], plan.in_shape[i], comm, plan.options);
    }

    executor_r2c = one_dim_backend<backend_tag>::make_r2c(plan.out_shape[0][this_mpi_rank], plan.fft_direction[0]);
    executor[0] = one_dim_backend<backend_tag>::make(plan.out_shape[1][this_mpi_rank], plan.fft_direction[1]);
    executor[1] = one_dim_backend<backend_tag>::make(plan.out_shape[2][this_mpi_rank], plan.fft_direction[2]);
}

template<typename backend_tag, typename index>
template<typename scalar_type>
void fft3d_r2c<backend_tag, index>::standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
                                                std::complex<scalar_type> workspace[], scale scaling) const{
    /*
     * Follows logic similar to the fft3d case but using directly the member shapers and executors.
     */
    int last = get_last_active(forward_shaper);

    scalar_type* reshaped_input = reinterpret_cast<scalar_type*>(workspace);
    scalar_type const *effective_input = input; // either input or the result of reshape operation 0
    if (forward_shaper[0]){
        add_trace name("reshape");
        forward_shaper[0]->apply(input, reshaped_input, reinterpret_cast<scalar_type*>(workspace + executor_r2c->real_size()));
        effective_input = reshaped_input;
    }

    if (last < 1){ // no reshapes after 0
        add_trace name("fft-1d x3");
        executor_r2c->forward(effective_input, output);
        executor[0]->forward(output);
        executor[1]->forward(output);
        apply_scale(direction::forward, scaling, output);
        return;
    }

    // if there is messier combination of transforms, then we need internal buffers
    std::complex<scalar_type>* temp_buffer = workspace + size_comm_buffers();
    { add_trace name("fft-1d r2c");
    executor_r2c->forward(effective_input, temp_buffer);
    }

    for(int i=1; i<last; i++){
        if (forward_shaper[i]){
            add_trace name("reshape");
            forward_shaper[i]->apply(temp_buffer, temp_buffer, workspace);
        }
        add_trace name("fft-1d");
        executor[i-1]->forward(temp_buffer);
    }
    { add_trace name("reshape");
    forward_shaper[last]->apply(temp_buffer, output, workspace);
    }

    for(int i=last-1; i<2; i++){
        add_trace name("fft-1d");
        executor[i]->forward(output);
    }

    apply_scale(direction::forward, scaling, output);
}

template<typename backend_tag, typename index>
template<typename scalar_type>
void fft3d_r2c<backend_tag, index>::standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
                                                std::complex<scalar_type> workspace[], scale scaling) const{
    /*
     * Follows logic similar to the fft3d case but using directly the member shapers and executors.
     */
    std::complex<scalar_type>* temp_buffer = workspace + size_comm_buffers();
    if (backward_shaper[0]){
        add_trace name("reshape");
        backward_shaper[0]->apply(input, temp_buffer, workspace);
    }else{
        add_trace name("copy");
        data_manipulator<location_tag>::copy_n(input, executor[1]->box_size(), temp_buffer);
    }

    for(int i=0; i<2; i++){ // apply the two complex-to-complex ffts
        { add_trace name("fft-1d");
        executor[1-i]->backward(temp_buffer);
        }
        if (backward_shaper[i+1]){
            add_trace name("reshape");
            backward_shaper[i+1]->apply(temp_buffer, temp_buffer, workspace);
        }
    }

    // the result of the first two ffts and three reshapes is stored in temp_buffer
    // executor 2 must apply complex to real backward transform
    if (backward_shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        scalar_type* real_buffer = reinterpret_cast<scalar_type*>(workspace);
        { add_trace name("fft-1d");
        executor_r2c->backward(temp_buffer, real_buffer);
        }
        add_trace name("reshape");
        backward_shaper[3]->apply(real_buffer, output, reinterpret_cast<scalar_type*>(workspace + executor_r2c->real_size()));
    }else{
        add_trace name("fft-1d");
        executor_r2c->backward(temp_buffer, output);
    }

    apply_scale(direction::backward, scaling, output);
}

#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_fft3d_r2c(backend::fftw, int)
heffte_instantiate_fft3d_r2c(backend::fftw, long long)
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_fft3d_r2c(backend::mkl, int)
heffte_instantiate_fft3d_r2c(backend::mkl, long long)
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_fft3d_r2c(backend::cufft, int)
heffte_instantiate_fft3d_r2c(backend::cufft, long long)
#endif
#ifdef Heffte_ENABLE_ROCM
heffte_instantiate_fft3d_r2c(backend::rocfft, int)
heffte_instantiate_fft3d_r2c(backend::rocfft, long long)
#endif

}
