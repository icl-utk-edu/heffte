/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_fft3d.h"

template<typename backend_tag>
void test_fft3d_r2c_const_dest2(MPI_Comm comm){
    assert(mpi::comm_size(comm) == 2);
    current_test<int, using_mpi, backend_tag> name("constructor heffte::fft3d_r2c", comm);
    box3d const world = {{0, 0, 0}, {4, 4, 4}};
    int const me = mpi::comm_rank(comm);

    for(int dim = 0; dim < 3; dim++){
        std::vector<box3d> rboxes = heffte::split_world(world, {2, 1, 1});
        std::vector<box3d> cboxes = heffte::split_world(world.r2c(dim), {2, 1, 1});
        // construct an instance of heffte::fft3d and delete it immediately
        heffte::fft3d_r2c<backend_tag> fft(rboxes[me], cboxes[me], dim, comm);
    }
}

template<typename backend_tag, typename scalar_type, int h0, int h1, int h2>
void test_fft3d_r2c_arrays(MPI_Comm comm){
    using output_type = typename fft_output<scalar_type>::type; // complex type of the output
    using input_container  = typename heffte::fft3d<backend_tag>::template buffer_container<scalar_type>; // std::vector or cuda::vector
    using output_container = typename heffte::fft3d<backend_tag>::template buffer_container<output_type>; // std::vector or cuda::vector

    // works with ranks 2 and 12 only
    int const num_ranks = mpi::comm_size(comm);
    assert(num_ranks == 1 or num_ranks == 2 or num_ranks == 12);
    current_test<scalar_type, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test heffte::fft3d_r2c", comm);

    double correction = 1.0; // single precision is less stable, especially for larger problems with 12 mpi ranks
    if (std::is_same<scalar_type, float>::value){
        correction = (num_ranks == 12) ? 1.E-4 : 1.E-2;
    }

    int const me = mpi::comm_rank(comm);
    box3d const rworld = {{0, 0, 0}, {h0, h1, h2}};
    auto world_input = make_data<scalar_type>(rworld);

    for(auto const &options : make_all_options<backend_tag>()){
    for(int dim = 0; dim < 3; dim++){
        box3d const cworld = rworld.r2c(dim);
        auto world_fft     = get_subbox(rworld, cworld, forward_fft<backend_tag>(rworld, world_input));

        for(int i=0; i<3; i++){
            // split the world into processors
            std::array<int, 3> split = {1, 1, 1};
            if (num_ranks == 2){
                split[i] = 2;
            }else if (num_ranks == 12){
                split = {2, 2, 2};
                split[i] = 3;
            }
            std::vector<box3d> rboxes = heffte::split_world(rworld, split);
            std::vector<box3d> cboxes = heffte::split_world(cworld, split);

            assert(rboxes.size() == num_ranks);
            assert(cboxes.size() == num_ranks);

            // get the local input as a cuda::vector or std::vector
            auto local_input = input_maker<backend_tag, scalar_type>::select(rworld, rboxes[me], world_input);
            auto reference_fft = get_subbox(cworld, cboxes[me], world_fft); // reference solution
            output_container forward(reference_fft.size()); // computed solution

            heffte::fft3d_r2c<backend_tag> fft(rboxes[me], cboxes[me], dim, comm, options);
            output_container workspace(fft.size_workspace());

            fft.forward(local_input.data(), forward.data(), workspace.data()); // compute the forward fft

            // compare to the reference
            tassert(approx(forward, reference_fft, correction));

            input_container backward(local_input.size()); // compute backward fft using scalar_type
            fft.backward(forward.data(), backward.data(), workspace.data());
            auto backward_result = rescale(rworld, backward, scale::full); // always std::vector
            tassert(approx(local_input, backward_result)); // compare with the original input
        }
    }
    } // different option variants
}

template<typename backend_tag, typename scalar_type, int h0, int h1, int h2>
void test_fft3d_r2c_vectors(MPI_Comm comm){
    // works with ranks 6 and 8 only
    int const num_ranks = mpi::comm_size(comm);
    assert(num_ranks == 6 or num_ranks == 8);
    current_test<scalar_type, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test heffte::fft3d_r2c", comm);

    double correction = 1.0; // single precision is less stable, especially for larger problems with 12 mpi ranks
    if (std::is_same<scalar_type, float>::value){
        //correction = (num_ranks == 6) ? 0.5E-4 : 1.E-2;
        correction = 1.0E-4;
    }

    int const me = mpi::comm_rank(comm);
    box3d const rworld = {{0, 0, 0}, {h0, h1, h2}};
    auto world_input = make_data<scalar_type>(rworld);

    std::array<heffte::scale, 3> fscale = {heffte::scale::none, heffte::scale::symmetric, heffte::scale::full};
    std::array<heffte::scale, 3> bscale = {heffte::scale::full, heffte::scale::symmetric, heffte::scale::none};

    for(auto const &options : make_all_options<backend_tag>()){
    for(int dim = 0; dim < 3; dim++){
        box3d const cworld = rworld.r2c(dim);
        auto world_fft     = get_subbox(rworld, cworld, forward_fft<backend_tag>(rworld, world_input));

        for(int i=0; i<3; i++){
            std::array<int, 3> split = {1, 1, 1};
            if (num_ranks == 6){
                split[i] = 2;
                split[(i+1) % 3] = 3;
            }else if (num_ranks == 8){
                split = {2, 2, 2};
            }
            std::vector<box3d> rboxes = heffte::split_world(rworld, split);
            std::vector<box3d> cboxes = heffte::split_world(cworld, split);

            assert(rboxes.size() == num_ranks);
            assert(cboxes.size() == num_ranks);

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

            box3d const inbox  = rboxes[iindex];
            box3d const outbox = cboxes[oindex];

            auto local_input   = input_maker<backend_tag, scalar_type>::select(rworld, inbox, world_input);
            auto reference_fft = rescale(rworld, get_subbox(cworld, outbox, world_fft), fscale[i]);

            heffte::fft3d_r2c<backend_tag> fft(inbox, outbox, dim, comm, options);

            auto result = fft.forward(local_input, fscale[i]);
            tassert(approx(result, reference_fft, correction));

            auto backward_result = fft.backward(result, bscale[i]);
            auto backward_scaled_result = rescale(rworld, backward_result, scale::none);
            tassert(approx(local_input, backward_scaled_result));
        }
    }
    } // different option variants
}

void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft_r2c class");
    int const num_ranks = mpi::comm_size(comm);
    int const me = mpi::comm_rank(comm);

    switch(num_ranks){
        case 1:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_arrays<backend::fftw, float, 3, 4, 5>(comm);
            test_fft3d_r2c_arrays<backend::fftw, double, 3, 4, 5>(comm);
            #endif
            #ifdef Heffte_ENABLE_MKL
            test_fft3d_r2c_arrays<backend::mkl, float, 3, 4, 5>(comm);
            test_fft3d_r2c_arrays<backend::mkl, double, 3, 4, 5>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_arrays<backend::cufft, float, 3, 4, 5>(comm);
            test_fft3d_r2c_arrays<backend::cufft, double, 3, 4, 5>(comm);
            #endif
            break;
        case 2:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_const_dest2<backend::fftw>(comm);
            test_fft3d_r2c_arrays<backend::fftw, float, 9, 9, 9>(comm);
            test_fft3d_r2c_arrays<backend::fftw, double, 9, 9, 9>(comm);
            #endif
            #ifdef Heffte_ENABLE_MKL
            test_fft3d_r2c_const_dest2<backend::mkl>(comm);
            test_fft3d_r2c_arrays<backend::mkl, float, 9, 9, 9>(comm);
            test_fft3d_r2c_arrays<backend::mkl, double, 9, 9, 9>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_const_dest2<backend::cufft>(comm);
            test_fft3d_r2c_arrays<backend::cufft, float, 9, 9, 9>(comm);
            test_fft3d_r2c_arrays<backend::cufft, double, 9, 9, 9>(comm);
            #endif
            break;
        case 6:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_vectors<backend::fftw, float, 11, 11, 20>(comm);
            test_fft3d_r2c_vectors<backend::fftw, double, 10, 10, 11>(comm);
            #endif
            #ifdef Heffte_ENABLE_MKL
            test_fft3d_r2c_vectors<backend::mkl, float, 11, 11, 20>(comm);
            test_fft3d_r2c_vectors<backend::mkl, double, 10, 10, 11>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_vectors<backend::cufft, float, 11, 11, 20>(comm);
            test_fft3d_r2c_vectors<backend::cufft, double, 10, 10, 11>(comm);
            #endif
            break;
        case 8:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_vectors<backend::fftw, float, 12, 12, 10>(comm);
            test_fft3d_r2c_vectors<backend::fftw, double, 15, 15, 18>(comm);
            #endif
            #ifdef Heffte_ENABLE_MKL
            test_fft3d_r2c_vectors<backend::mkl, float, 12, 12, 10>(comm);
            test_fft3d_r2c_vectors<backend::mkl, double, 15, 15, 18>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_vectors<backend::cufft, float, 12, 12, 10>(comm);
            test_fft3d_r2c_vectors<backend::cufft, double, 15, 15, 18>(comm);
            #endif
            break;
        case 12:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_arrays<backend::fftw, float, 21, 20, 20>(comm);
            test_fft3d_r2c_arrays<backend::fftw, double, 20, 20, 19>(comm);
            #endif
            #ifdef Heffte_ENABLE_MKL
            test_fft3d_r2c_arrays<backend::mkl, float, 21, 20, 20>(comm);
            test_fft3d_r2c_arrays<backend::mkl, double, 20, 20, 19>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_arrays<backend::cufft, float, 21, 20, 20>(comm);
            test_fft3d_r2c_arrays<backend::cufft, double, 20, 20, 19>(comm);
            #endif
            break;
        default:
            throw std::runtime_error("No test for the given number of ranks!");
    };
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
