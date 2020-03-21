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
    template void fft3d_r2c<some_backend>::standard_transform<float>(float const[], std::complex<float>[], scale) const;    \
    template void fft3d_r2c<some_backend>::standard_transform<double>(double const[], std::complex<double>[], scale) const; \
    template void fft3d_r2c<some_backend>::standard_transform<float>(std::complex<float> const[], float[], scale) const;    \
    template void fft3d_r2c<some_backend>::standard_transform<double>(std::complex<double> const[], double[], scale) const; \

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
    fft_r2c = one_dim_backend<backend_tag>::make_r2c(shape0[me], dirs[0]);
    fft1 = one_dim_backend<backend_tag>::make(shape1[me], dirs[1]);
    fft2 = one_dim_backend<backend_tag>::make(shape2[me], dirs[2]);
}

template<typename backend_tag>
template<typename scalar_type>
void fft3d_r2c<backend_tag>::standard_transform(scalar_type const input[], std::complex<scalar_type> output[], scale scaling) const{
    /*
     * Follows logic similar to the fft3d case but using directly the member shapers and executors.
     */
    buffer_container<scalar_type> reshaped_input;
    scalar_type const *effective_input = input; // either input or the result of reshape operation 0
    if (forward_shaper[0]){
        reshaped_input = buffer_container<scalar_type>(fft_r2c->real_size());
        forward_shaper[0]->apply(input, reshaped_input.data());
        effective_input = reshaped_input.data();
    }

    if (not forward_shaper[1] and not forward_shaper[2] and not forward_shaper[3]){
        fft_r2c->forward(effective_input, output);
        fft1->forward(output);
        fft2->forward(output);
        return;
    }

    // if there is messier combination of transforms, then we need internal buffers
    size_t buffer_size = std::max(std::max(fft_r2c->complex_size(), fft1->box_size()), fft2->box_size());
    buffer_container<std::complex<scalar_type>> buff0(buffer_size);

    fft_r2c->forward(effective_input, buff0.data());
    reshaped_input = buffer_container<scalar_type>(); // release the temporary real data (if any)

    // the second buffer is needed only if two of the reshape operations are active
    buffer_container<std::complex<scalar_type>> buff1(
                            (!!forward_shaper[2] or !!forward_shaper[3]) ? buffer_size : 0
                                           );
    std::complex<scalar_type> *data = buff0.data();
    std::complex<scalar_type> *temp = buff1.data();

    reshape_stage(forward_shaper[1], data, (!!forward_shaper[2] or !!forward_shaper[3]) ? temp : output);
    fft1->forward(data);

    reshape_stage(forward_shaper[2], data, (!!forward_shaper[3]) ? temp : output);
    fft2->forward(data);

    reshape_stage(forward_shaper[3], data, output);

    if (scaling != scale::none)
        data_scaling<typename backend::buffer_traits<backend_tag>::location>::apply(
            size_outbox(), data,
            (scaling == scale::full) ? scale_factor : std::sqrt(scale_factor));
}

template<typename backend_tag>
template<typename scalar_type>
void fft3d_r2c<backend_tag>::standard_transform(std::complex<scalar_type> const input[], scalar_type output[], scale scaling) const{
    /*
     * Follows logic similar to the fft3d case but using directly the member shapers and executors.
     */
    // we need to know the size of the internal buffers
    size_t buffer_size = std::max(std::max(fft_r2c->complex_size(), fft1->box_size()), fft2->box_size());

    // perform stage 0 transformation
    buffer_container<std::complex<scalar_type>> buff0;
    if (backward_shaper[0]){
        buff0 = buffer_container<std::complex<scalar_type>>(buffer_size);
        backward_shaper[0]->apply(input, buff0.data());
    }else{
        buff0 = buffer_container<std::complex<scalar_type>>(input, input + fft2->box_size());
    }

    fft2->backward(buff0.data()); // first fft of the backward transform

    buffer_container<std::complex<scalar_type>> buff1;
    if (backward_shaper[1] and backward_shaper[2]){
        // will do two reshape, the data will move buff0 -> buff1 -> buff0
        buff1 = buffer_container<std::complex<scalar_type>>(buffer_size);
        backward_shaper[1]->apply(buff0.data(), buff1.data());
        fft1->backward(buff1.data());
        backward_shaper[2]->apply(buff1.data(), buff0.data());
    }else if (backward_shaper[1] or backward_shaper[2]){
        // will do only one reshape, the data will move buff0 -> buff1
        // and the two containers need to be swapped
        buff1 = buffer_container<std::complex<scalar_type>>(buffer_size);
        if (backward_shaper[1]){
            backward_shaper[1]->apply(buff0.data(), buff1.data());
            fft1->backward(buff1.data());
        }else{
            fft1->backward(buff0.data());
            backward_shaper[2]->apply(buff0.data(), buff1.data());
        }
        std::swap(buff0, buff1);
    }
    buff1 = buffer_container<std::complex<scalar_type>>(); // clear the buffer

    // the result of the first two ffts and three reshapes is stored in buff0
    // executor 2 must apply complex to real backward transform
    if (backward_shaper[3]){
        // there is one more reshape left, transform into a real temporary buffer
        buffer_container<scalar_type> real_buffer(fft_r2c->real_size());
        fft_r2c->backward(buff0.data(), real_buffer.data());
        backward_shaper[3]->apply(real_buffer.data(), output);
    }else{
        fft_r2c->backward(buff0.data(), output);
    }

    if (scaling != scale::none)
        data_scaling<typename backend::buffer_traits<backend_tag>::location>::apply(
            size_inbox(), output,
            (scaling == scale::full) ? scale_factor : std::sqrt(scale_factor));
}

#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_fft3d_r2c(backend::fftw);
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_fft3d_r2c(backend::cufft);
#endif

}
