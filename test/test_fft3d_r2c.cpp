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
    assert(num_ranks == 2 or num_ranks == 12);
    current_test<scalar_type, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test heffte::fft3d_r2c", comm);

    int const me = mpi::comm_rank(comm);
    box3d const rworld = {{0, 0, 0}, {h0, h1, h2}};
    auto world_input = make_data<scalar_type>(rworld);

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

            heffte::fft3d_r2c<backend_tag> fft(rboxes[me], cboxes[me], dim, comm);

            fft.forward(local_input.data(), forward.data()); // compute the forward fft

            // compare to the reference
            if (std::is_same<scalar_type, float>::value){
                tassert(approx(forward, reference_fft, 0.01)); // float gives error 5.E-5, correct to drop below 1.E-6
            }else{
                tassert(approx(forward, reference_fft));
            }

            input_container backward(local_input.size()); // compute backward fft using scalar_type
            fft.backward(forward.data(), backward.data());
            auto backward_result = rescale(rworld, backward, scale::full); // always std::vector
            tassert(approx(local_input, backward_result)); // compare with the original input
        }
    }
}

void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft_r2c class");
    int const num_ranks = mpi::comm_size(comm);
    int const me = mpi::comm_rank(comm);

    switch(num_ranks){
        case 2:
            #ifdef Heffte_ENABLE_FFTW
            test_fft3d_r2c_const_dest2<backend::fftw>(comm);
            test_fft3d_r2c_arrays<backend::fftw, float, 9, 9, 9>(comm);
            test_fft3d_r2c_arrays<backend::fftw, double, 9, 9, 9>(comm);
            #endif
            #ifdef Heffte_ENABLE_CUDA
            test_fft3d_r2c_const_dest2<backend::cufft>(comm);
            test_fft3d_r2c_arrays<backend::cufft, float, 9, 9, 9>(comm);
            test_fft3d_r2c_arrays<backend::cufft, double, 9, 9, 9>(comm);
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
