/**
 * @class
 * CPU functions of HEFFT
 */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/


#include "heffte_fft3d_r2c.h"

#define heffte_instantiate_fft3d_r2c(some_backend) \
    template class fft3d_r2c<some_backend>; \
    template void fft3d_r2c<some_backend>::standard_transform<float>(float const[], std::complex<float>[], std::complex<float>[], scale) const;    \
    template void fft3d_r2c<some_backend>::standard_transform<double>(double const[], std::complex<double>[], std::complex<double>[], scale) const; \
    template void fft3d_r2c<some_backend>::standard_transform<float>(std::complex<float> const[], float[], std::complex<float>[], scale) const;    \
    template void fft3d_r2c<some_backend>::standard_transform<double>(std::complex<double> const[], double[], std::complex<double>[], scale) const; \

namespace heffte {

template<typename backend_tag>
fft3d_r2c<backend_tag>::fft3d_r2c(logic_plan3d const &plan, int const this_mpi_rank, MPI_Comm const comm) :
    pinbox(new box3d(plan.in_shape[0][this_mpi_rank])), poutbox(new box3d(plan.out_shape[3][this_mpi_rank])),
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

template<typename backend_tag>
template<typename scalar_type>
void fft3d_r2c<backend_tag>::standard_transform(scalar_type const input[], std::complex<scalar_type> output[],
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

    if (scaling != scale::none){
        add_trace name("scale");
        data_manipulator<location_tag>::scale(size_outbox(), output, get_scale_factor(scaling));
    }
}

template<typename backend_tag>
template<typename scalar_type>
void fft3d_r2c<backend_tag>::standard_transform(std::complex<scalar_type> const input[], scalar_type output[],
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

    if (scaling != scale::none){
        add_trace name("scale");
        data_manipulator<location_tag>::scale(size_inbox(), output, get_scale_factor(scaling));
    }
}

#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_fft3d_r2c(backend::fftw);
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_fft3d_r2c(backend::mkl);
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_fft3d_r2c(backend::cufft);
#endif

}
