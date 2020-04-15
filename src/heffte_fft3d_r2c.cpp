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
    pinbox(plan.in_shape[0][this_mpi_rank]), poutbox(plan.out_shape[3][this_mpi_rank]),
    scale_factor(1.0 / static_cast<double>(plan.index_count))
{
    for(int i=0; i<4; i++){
        forward_shaper[i]    = make_reshape3d_alltoallv<backend_tag>(plan.in_shape[i], plan.out_shape[i], comm);
        backward_shaper[3-i] = make_reshape3d_alltoallv<backend_tag>(plan.out_shape[i], plan.in_shape[i], comm);
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

    buffer_container<scalar_type> reshaped_input;
    scalar_type const *effective_input = input; // either input or the result of reshape operation 0
    if (forward_shaper[0]){
        reshaped_input = buffer_container<scalar_type>(executor_r2c->real_size());
        forward_shaper[0]->apply(input, reshaped_input.data(), reinterpret_cast<scalar_type*>(workspace));
        effective_input = reshaped_input.data();
    }

    if (last < 1){ // no reshapes after 0
        add_trace name("fft-1d x3");
        executor_r2c->forward(effective_input, output);
        executor[0]->forward(output);
        executor[1]->forward(output);
        return;
    }

    // if there is messier combination of transforms, then we need internal buffers
    size_t buffer_size = get_max_size(executor_r2c, executor);
    buffer_container<std::complex<scalar_type>> temp_buffer(buffer_size);
    executor_r2c->forward(effective_input, temp_buffer.data());
    reshaped_input = buffer_container<scalar_type>();

    for(int i=1; i<last; i++){
        if (forward_shaper[i])
            forward_shaper[i]->apply(temp_buffer.data(), temp_buffer.data(), workspace);
        add_trace name("fft-1d");
        executor[i-1]->forward(temp_buffer.data());
    }
    forward_shaper[last]->apply(temp_buffer.data(), output, workspace);

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
    size_t buffer_size = get_max_size(executor_r2c, executor);
    buffer_container<std::complex<scalar_type>> temp_buffer(buffer_size);
    if (backward_shaper[0]){
        backward_shaper[0]->apply(input, temp_buffer.data(), workspace);
    }else{
        data_manipulator<location_tag>::copy_n(input, executor[1]->box_size(), temp_buffer.data());
    }

    for(int i=0; i<2; i++){ // apply the two complex-to-complex ffts
        { add_trace name("fft-1d");
        executor[1-i]->backward(temp_buffer.data());
        }
        if (backward_shaper[i+1])
            backward_shaper[i+1]->apply(temp_buffer.data(), temp_buffer.data(), workspace);
    }

    // the result of the first two ffts and three reshapes is stored in temp_buffer
    // executor 2 must apply complex to real backward transform
    if (backward_shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        buffer_container<scalar_type> real_buffer(executor_r2c->real_size());
        { add_trace name("fft-1d");
        executor_r2c->backward(temp_buffer.data(), real_buffer.data());
        }
        temp_buffer = buffer_container<std::complex<scalar_type>>(); // clean temp_buffer
        backward_shaper[3]->apply(real_buffer.data(), output, reinterpret_cast<scalar_type*>(workspace));
    }else{
        add_trace name("fft-1d");
        executor_r2c->backward(temp_buffer.data(), output);
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
