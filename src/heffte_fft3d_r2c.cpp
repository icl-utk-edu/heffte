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
fft3d_r2c<backend_tag>::fft3d_r2c(box3d const cinbox, box3d const coutbox, int r2c_direction, MPI_Comm const comm) : inbox(cinbox), outbox(coutbox){
    static_assert(backend::is_enabled<backend_tag>::value, "Requested backend is invalid or has not been enabled.");

    // assemble the entire box layout first
    // perform all analysis for all reshape operation without further communication
    // create the reshape objects
    ioboxes boxes = mpi::gather_boxes(inbox, outbox, comm);
    box3d const real_world = find_world(boxes.in);
    box3d const complex_world = find_world(boxes.out);

    assert(real_world.r2c(r2c_direction) == complex_world);
    assert( world_complete(boxes.in, real_world) );
    assert( world_complete(boxes.out, complex_world) );

    scale_factor = 1.0 / static_cast<double>(real_world.count());

    std::array<int, 2> proc_grid = make_procgrid(mpi::comm_size(comm));

    // the directions in which to do transformations
    std::array<int, 3> dirs = {r2c_direction, (r2c_direction + 1) % 3, (r2c_direction + 2) % 3};

    std::vector<box3d> shape0 = make_pencils(real_world, proc_grid, dirs[0], boxes.in);
    std::vector<box3d> shape0_r2c;
    for(auto b : shape0) shape0_r2c.push_back(b.r2c(r2c_direction));
    std::vector<box3d> shape1 = make_pencils(complex_world, proc_grid, dirs[1], shape0_r2c);
    std::vector<box3d> shape2 = make_pencils(complex_world, proc_grid, dirs[2], shape1);

    forward_shaper[0] = make_reshape3d_alltoallv<backend_tag>(boxes.in, shape0, comm);
    forward_shaper[1] = make_reshape3d_alltoallv<backend_tag>(shape0_r2c, shape1, comm);
    forward_shaper[2] = make_reshape3d_alltoallv<backend_tag>(shape1, shape2, comm);
    forward_shaper[3] = make_reshape3d_alltoallv<backend_tag>(shape2, boxes.out, comm);

    backward_shaper[0] = make_reshape3d_alltoallv<backend_tag>(boxes.out, shape2, comm);
    backward_shaper[1] = make_reshape3d_alltoallv<backend_tag>(shape2, shape1, comm);
    backward_shaper[2] = make_reshape3d_alltoallv<backend_tag>(shape1, shape0_r2c, comm);
    backward_shaper[3] = make_reshape3d_alltoallv<backend_tag>(shape0, boxes.in, comm);

    int const me = mpi::comm_rank(comm);
    executor_r2c = one_dim_backend<backend_tag>::make_r2c(shape0[me], dirs[0]);
    executor[0] = one_dim_backend<backend_tag>::make(shape1[me], dirs[1]);
    executor[1] = one_dim_backend<backend_tag>::make(shape2[me], dirs[2]);
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
        executor[i-1]->forward(temp_buffer.data());
    }
    forward_shaper[last]->apply(temp_buffer.data(), output, workspace);

    for(int i=last-1; i<2; i++)
        executor[i]->forward(output);

    if (scaling != scale::none)
        data_manipulator<location_tag>::scale(size_outbox(), output, get_scale_factor(scaling));
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
        data_manipulator<location_tag>::copy_n(input, executor[0]->box_size(), temp_buffer.data());
    }

    for(int i=0; i<2; i++){ // apply the two complex-to-complex ffts
        executor[1-i]->backward(temp_buffer.data());
        if (backward_shaper[i+1])
            backward_shaper[i+1]->apply(temp_buffer.data(), temp_buffer.data(), workspace);
    }

    // the result of the first two ffts and three reshapes is stored in temp_buffer
    // executor 2 must apply complex to real backward transform
    if (backward_shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        buffer_container<scalar_type> real_buffer(executor_r2c->real_size());
        executor_r2c->backward(temp_buffer.data(), real_buffer.data());
        temp_buffer = buffer_container<std::complex<scalar_type>>(); // clean temp_buffer
        backward_shaper[3]->apply(real_buffer.data(), output, reinterpret_cast<scalar_type*>(workspace));
    }else{
        executor_r2c->backward(temp_buffer.data(), output);
    }

    if (scaling != scale::none)
        data_manipulator<location_tag>::scale(size_inbox(), output, get_scale_factor(scaling));
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
