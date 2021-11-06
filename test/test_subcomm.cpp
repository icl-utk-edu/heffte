/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_fft3d.h"

template<typename backend_tag>
void test_subcomm_cases(MPI_Comm const comm){
    using location_tag = typename backend::buffer_traits<backend_tag>::location;
    using input_type  = double;
    using cinput_type = std::complex<double>;
    using output_type = std::complex<double>;
    //using output_container  = typename heffte::fft3d<backend_tag>::template buffer_container<double>;
    //using coutput_container = typename heffte::fft3d<backend_tag>::template buffer_container<std::complex<double>>;

    int const me        = mpi::comm_rank(comm);
    int const num_ranks = mpi::comm_size(comm);

    current_test<double, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test subcommunicator", comm);

    box3d<> const world = {{0, 0, 0}, {23, 23, 23}};

    std::array<int,3> proc_i = heffte::proc_setup_min_surface(world, num_ranks);

    std::vector<box3d<int>> inboxes  = heffte::split_world(world, proc_i);

    auto world_input = make_data<input_type>(world);
    auto world_fft = forward_fft<backend_tag>(world, world_input);

    auto cworld_input = make_data<output_type>(world);
    auto cworld_fft   = forward_fft<backend_tag>(world, cworld_input);

    auto local_input  = input_maker<backend_tag, input_type>::select(world, inboxes[me], world_input);
    auto clocal_input = input_maker<backend_tag, cinput_type>::select(world, inboxes[me], cworld_input);

    auto local_ref   = get_subbox(world, inboxes[me], world_fft);
    auto clocal_ref  = get_subbox(world, inboxes[me], cworld_fft);

    backend::device_instance<location_tag> device;

    for(auto const &num_subcomm : std::vector<int>{1, 2, 3, 4}){
        for(int variant=0; variant<4; variant++){
            for(auto const &alg : std::vector<reshape_algorithm>{
                reshape_algorithm::alltoall, reshape_algorithm::alltoallv,
                reshape_algorithm::p2p, reshape_algorithm::p2p_plined}){

                heffte::plan_options options = default_options<backend_tag>();

                options.use_subcomm(num_subcomm);
                options.use_pencils = (variant / 2 == 0);
                options.use_reorder = (variant % 2 == 0);
                options.algorithm = alg;

                auto fft = make_fft3d<backend_tag>(inboxes[me], inboxes[me], comm, options);

                auto lresult = make_buffer_container<output_type>(device.stream(), fft.size_outbox());
                auto lback   = make_buffer_container<input_type>(device.stream(), fft.size_inbox());
                auto clback  = make_buffer_container<output_type>(device.stream(), fft.size_inbox());

                fft.forward(local_input.data(), lresult.data());
                tassert(approx(lresult, local_ref));

                fft.backward(lresult.data(), lback.data(), heffte::scale::full);
                tassert(approx(local_input, lback));

                fft.forward(clocal_input.data(), lresult.data());
                tassert(approx(lresult, clocal_ref));

                fft.backward(lresult.data(), clback.data(), heffte::scale::full);
                tassert(approx(clocal_input, clback));
            }
        }
    }
}
template<typename backend_tag>
void test_subcomm_cases_r2c(MPI_Comm const comm){
    using location_tag = typename backend::buffer_traits<backend_tag>::location;
    using input_type  = float;
    using output_type = std::complex<float>;

    int const me        = mpi::comm_rank(comm);
    int const num_ranks = mpi::comm_size(comm);

    current_test<double, using_mpi, backend_tag> name(std::string("-np ") + std::to_string(num_ranks) + "  test subcommunicator", comm);

    box3d<> const world = {{0, 0, 0}, {23, 23, 23}};

    std::array<int,3> proc_i = heffte::proc_setup_min_surface(world, num_ranks);

    std::vector<box3d<int>> inboxes  = heffte::split_world(world, proc_i);

    auto world_input = make_data<input_type>(world);
    auto world_fft = forward_fft<backend_tag>(world, world_input);


    auto local_input  = input_maker<backend_tag, input_type>::select(world, inboxes[me], world_input);

    backend::device_instance<location_tag> device;

    for(auto const &num_subcomm : std::vector<int>{1, 2, 4}){
        for(int dim=0; dim<3; dim++){
            std::vector<box3d<>> cboxes = heffte::split_world(world.r2c(dim), proc_i);
            auto world_fft_r2c = get_subbox(world, world.r2c(dim), world_fft);
            auto local_ref = get_subbox(world.r2c(dim), cboxes[me], world_fft_r2c);

        for(int variant=0; variant<4; variant++){
            for(auto const &alg : std::vector<reshape_algorithm>{
                reshape_algorithm::alltoall, reshape_algorithm::alltoallv,
                reshape_algorithm::p2p, reshape_algorithm::p2p_plined}){

                heffte::plan_options options = default_options<backend_tag>();

                options.use_subcomm(num_subcomm);
                options.use_pencils = (variant / 2 == 0);
                options.use_reorder = (variant % 2 == 0);
                options.algorithm = alg;

                auto fft = make_fft3d_r2c<backend_tag>(inboxes[me], cboxes[me], dim, comm, options);

                auto lresult = make_buffer_container<output_type>(device.stream(), fft.size_outbox());
                auto lback   = make_buffer_container<input_type>(device.stream(), fft.size_inbox());

                fft.forward(local_input.data(), lresult.data());
                tassert(approx(lresult, local_ref));

                fft.backward(lresult.data(), lback.data(), heffte::scale::full);
                tassert(approx(local_input, lback));
            }
        }
        }
    }
}

void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft subcommunicators");

    test_subcomm_cases<backend::stock>(comm);
    test_subcomm_cases_r2c<backend::stock>(comm);
    #ifdef Heffte_ENABLE_FFTW
    test_subcomm_cases<backend::fftw>(comm);
    test_subcomm_cases_r2c<backend::fftw>(comm);
    #endif
    #ifdef Heffte_ENABLE_MKL
    test_subcomm_cases<backend::mkl>(comm);
    test_subcomm_cases_r2c<backend::mkl>(comm);
    #endif
    #ifdef Heffte_ENABLE_CUDA
    test_subcomm_cases<backend::cufft>(comm);
    test_subcomm_cases_r2c<backend::cufft>(comm);
    #endif
    #ifdef Heffte_ENABLE_ROCM
    test_subcomm_cases<backend::rocfft>(comm);
    test_subcomm_cases_r2c<backend::rocfft>(comm);
    #endif
    #ifdef Heffte_ENABLE_ONEAPI
    test_subcomm_cases<backend::onemkl>(comm);
    test_subcomm_cases_r2c<backend::onemkl>(comm);
    #endif
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
