/** @class */
/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

#ifdef Heffte_ENABLE_FFTW
using default_cpu_backend = heffte::backend::fftw;
#elif defined(Heffte_ENABLE_MKL)
using default_cpu_backend = heffte::backend::mkl;
#endif

#if defined(Heffte_ENABLE_FFTW) || defined(Heffte_ENABLE_MKL)
#define HAS_CPU_BACKEND
#endif

/*
 * Simple unit test that checks the operation that gathers boxes across an mpi comm.
 */
void test_boxes(MPI_Comm const comm){
    current_test<> test("heffte::mpi::gather_boxes", comm);

    int const me = mpi::comm_rank(comm);

    std::vector<box3d> reference_inboxes;
    std::vector<box3d> reference_outboxes;

    for(int i=0; i<mpi::comm_size(comm); i++){
        reference_inboxes.push_back({{i, i+1, i+2}, {i+3, i+4, i+5}});
        reference_outboxes.push_back({{i, i+3, i+5}, {i+7, i+6, i+9}});
    }

    ioboxes boxes = mpi::gather_boxes(reference_inboxes[me], reference_outboxes[me], comm);

    tassert(match(boxes.in,  reference_inboxes));
    tassert(match(boxes.out, reference_outboxes));
}

/*
 * Returns a vector of data corresponding to a sub-box of the original world.
 * The entries are floating point numbers (real or complex) but have integer values
 * corresponding to the indexes in the world box.
 * Thus, by checking the indexes, it is easy to check if data was moved correctly
 * from one sub-box to another.
 */
template<typename scalar_type>
std::vector<scalar_type> get_subdata(box3d const world, box3d const subbox){
    // the entries in the master box go 0, 1, 2, 3, 4 ...
    int const wmidstride  = world.size[0];
    int const wslowstride = world.size[0] * world.size[1];
    int const smidstride  = subbox.size[0];
    int const sslowstride = subbox.size[0] * subbox.size[1];

    std::vector<scalar_type> result(subbox.count());

    for(int k = 0; k < subbox.size[2]; k++){
        for(int j = 0; j < subbox.size[1]; j++){
            for(int i = 0; i < subbox.size[0]; i++){
                result[k * sslowstride + j * smidstride + i]
                    = static_cast<scalar_type>((k + world.low[2] + subbox.low[2]) * wslowstride
                                                + (j + world.low[1] + subbox.low[1]) * wmidstride
                                                + i + world.low[0] + subbox.low[0]);
            }
        }
    }
    return result;
}

template<typename backend_tag, typename variant_tag>
std::unique_ptr<reshape3d_base>
make_test_reshape3d(std::vector<box3d> const &input_boxes, std::vector<box3d> const &output_boxes, MPI_Comm const comm){
    if (std::is_same<variant_tag, using_alltoall>::value){
        return make_reshape3d_alltoallv<backend_tag>(input_boxes, output_boxes, comm);
    }else{
        return make_reshape3d_pointtopoint<backend_tag>(input_boxes, output_boxes, comm);
    }
}

// splits the world box into a set of boxes with gird given by proc_grid
template<int hfast, int hmid, int hslow, int pfast, int pmid, int pslow, typename scalar_type, typename backend_tag, typename variant>
void test_cpu(MPI_Comm const comm){
    /*
     * simple test, create a world of indexes going all the way to hfast, hmid and hslow
     * then split the world into boxes numbering pfast, pmid, and pslow, assume that's what each rank owns
     * then create a new world of pencils and assigns a pencil to each rank (see the shuffle comment)
     * more the data from the original configuration to the new and check against reference data
     */
    current_test<scalar_type, using_mpi, backend_tag> test("-np " + std::to_string(mpi::comm_size(comm)) + "  "
                                                           + get_description<variant>(), comm);
    tassert( pfast * pmid * pslow == heffte::mpi::comm_size(comm) );

    int const me = heffte::mpi::comm_rank(comm);
    int const shift = 3;

    box3d world = {{0, 0, 0}, {hfast, hmid, hslow}};

    auto boxes   = split_world(world, {pfast, pmid, pslow});
    auto pencils = split_world(world, {pfast,    1, pmid * pslow});

    std::vector<box3d> rotate_boxes;
    if (std::is_same<scalar_type, std::complex<float>>::value){
        // shuffle the pencil boxes in some tests to check the case when there is no overlap between inbox and outbox
        // for the 2 by 2 grid, this shuffle ensures no overlap
        for(size_t i=0; i<boxes.size(); i++) rotate_boxes.push_back( pencils[(i + shift) % boxes.size()] );
    }else{
        for(auto b : pencils) rotate_boxes.push_back(b);
    }

    // create caches for a reshape algorithm, including creating a new mpi comm
    auto reshape = make_test_reshape3d<backend_tag, variant>(boxes, rotate_boxes, comm);
    std::vector<scalar_type> workspace(reshape->size_workspace());

    auto input_data     = get_subdata<scalar_type>(world, boxes[me]);
    auto reference_data = get_subdata<scalar_type>(world, rotate_boxes[me]);
    auto output_data    = std::vector<scalar_type>(rotate_boxes[me].count());

    if (std::is_same<scalar_type, float>::value){
        // sometimes, run two tests to make sure there is no internal corruption
        // there is no need to do that for every data type
        reshape->apply(input_data.data(), output_data.data(), workspace.data());
        output_data = std::vector<scalar_type>(rotate_boxes[me].count());
        reshape->apply(input_data.data(), output_data.data(), workspace.data());
    }else{
        reshape->apply(input_data.data(), output_data.data(), workspace.data());
    }

//     mpi::dump(0, input_data,     "input");
//     mpi::dump(0, output_data,    "output");
//     mpi::dump(0, reference_data, "reference");

    tassert(match(output_data, reference_data));
}

#ifdef Heffte_ENABLE_GPU
// splits the world box into a set of boxes with gird given by proc_grid
template<int hfast, int hmid, int hslow, int pfast, int pmid, int pslow, typename scalar_type, typename backend_tag, typename variant>
void test_gpu(MPI_Comm const comm){
    /*
     * similar to the CPU case, but the data is located on the GPU
     */
    current_test<scalar_type, using_mpi, backend_tag> test("-np " + std::to_string(mpi::comm_size(comm)) + "  "
                                                           + get_description<variant>(), comm);
    tassert( pfast * pmid * pslow == heffte::mpi::comm_size(comm) );

    int const me = heffte::mpi::comm_rank(comm);
    int const shift = 3;

    box3d world = {{0, 0, 0}, {hfast, hmid, hslow}};

    auto boxes   = split_world(world, {pfast, pmid, pslow});
    auto pencils = split_world(world, {pfast,    1, pmid * pslow});

    std::vector<box3d> rotate_boxes;
    if (std::is_same<scalar_type, std::complex<float>>::value){
        // shuffle the pencil boxes in some tests to check the case when there is no overlap between inbox and outbox
        // for the 2 by 2 grid, this shuffle ensures no overlap
        for(size_t i=0; i<boxes.size(); i++) rotate_boxes.push_back( pencils[(i + shift) % boxes.size()] );
    }else{
        for(auto b : pencils) rotate_boxes.push_back(b);
    }

    // create caches for a reshape algorithm, including creating a new mpi comm
    auto reshape = make_test_reshape3d<backend_tag, variant>(boxes, rotate_boxes, comm);
    gpu::vector<scalar_type> workspace(reshape->size_workspace());

    auto input_data     = get_subdata<scalar_type>(world, boxes[me]);
    auto cuinput_data   = gpu::transfer::load(input_data);
    auto reference_data = get_subdata<scalar_type>(world, rotate_boxes[me]);
    auto output_data    = gpu::vector<scalar_type>(rotate_boxes[me].count());

    if (std::is_same<scalar_type, float>::value){
        // sometimes, run two tests to make sure there is no internal corruption
        // there is no need to do that for every data type
        reshape->apply(cuinput_data.data(), output_data.data(), workspace.data());
        output_data = gpu::vector<scalar_type>(rotate_boxes[me].count());
        reshape->apply(cuinput_data.data(), output_data.data(), workspace.data());
    }else{
        reshape->apply(cuinput_data.data(), output_data.data(), workspace.data());
    }

    //auto ramout = cuda::unload(output_data);
    //mpi::dump(3, reference_data,     "reference_data");
    //mpi::dump(3, ramout,             "ramout");

    tassert(match(output_data, reference_data));
}
#endif

template<typename scalar_type>
void test_direct_reordered(MPI_Comm const comm){
    assert(mpi::comm_size(comm) == 4); // the rest is designed for 4 ranks
    current_test<scalar_type> test("-np " + std::to_string(mpi::comm_size(comm)) + "  direct_packer (unordered)", comm);

    box3d ordered_world({0, 0, 0}, {8, 9, 10});

    auto ordered_inboxes  = split_world(ordered_world, {1, 2, 2});
    auto ordered_outboxes = split_world(ordered_world, {2, 2, 1});

    int const me = heffte::mpi::comm_rank(comm);
    auto input     = get_subdata<scalar_type>(ordered_world, ordered_inboxes[me]);
    auto reference = get_subdata<scalar_type>(ordered_world, ordered_outboxes[me]);

    box3d world({0, 0, 0}, {9, 10, 8}, {2, 0, 1});

    std::vector<box3d> inboxes, outboxes;
    for(auto b : split_world(world, {2, 2, 1})) inboxes.push_back({b.low, b.high, world.order});
    std::vector<box3d> temp = split_world(world, {2, 1, 2}); // need to swap the middle two entries
    for(auto i : std::vector<int>{0, 2, 1, 3}) outboxes.push_back({temp[i].low, temp[i].high, world.order});

    #ifdef Heffte_ENABLE_FFTW
    {
        auto reshape = make_reshape3d_alltoallv<backend::fftw>(inboxes, outboxes, comm);
        std::vector<scalar_type> result(ordered_outboxes[me].count());
        std::vector<scalar_type> workspace(reshape->size_workspace());
        reshape->apply(input.data(), result.data(), workspace.data());

        tassert(match(result, reference));
    }{
        auto reshape = make_reshape3d_pointtopoint<backend::fftw>(inboxes, outboxes, comm);
        std::vector<scalar_type> result(ordered_outboxes[me].count());
        std::vector<scalar_type> workspace(reshape->size_workspace());
        reshape->apply(input.data(), result.data(), workspace.data());

        tassert(match(result, reference));
    }
    #endif

    #ifdef Heffte_ENABLE_GPU
    {
        auto reshape = make_reshape3d_alltoallv<gpu_backend>(inboxes, outboxes, comm);
        gpu::vector<scalar_type> workspace(reshape->size_workspace());

        auto cuinput = gpu::transfer::load(input);
        gpu::vector<scalar_type> curesult(ordered_outboxes[me].count());

        reshape->apply(cuinput.data(), curesult.data(), workspace.data());

        tassert(match(curesult, reference));
    }{
        auto reshape = make_reshape3d_pointtopoint<gpu_backend>(inboxes, outboxes, comm);
        gpu::vector<scalar_type> workspace(reshape->size_workspace());

        auto cuinput = gpu::transfer::load(input);
        gpu::vector<scalar_type> curesult(ordered_outboxes[me].count());

        reshape->apply(cuinput.data(), curesult.data(), workspace.data());
        tassert(match(curesult, reference));
    }
    #endif
}

template<typename scalar_type, typename variant>
void test_reshape_transposed(MPI_Comm comm){
    assert(mpi::comm_size(comm) == 4); // the rest is designed for 4 ranks
    current_test<scalar_type> test("-np " + std::to_string(mpi::comm_size(comm)) + "  " + get_description<variant>() + " transposed", comm);

    box3d world({0, 0, 0}, {1, 2, 3});

    std::vector<box3d> inboxes = split_world(world, {2, 1, 2});
    std::vector<box3d> ordered_outboxes = split_world(world, {1, 1, 4});

    int const me = heffte::mpi::comm_rank(comm);
    std::vector<scalar_type> input = get_subdata<scalar_type>(world, inboxes[me]);


    // the idea of the test is to try all combinations of order
    // test that MPI reshape with direct packer + on-node transpose reshape is equivalent to MPI reshape with transpose packer
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if (i != j){
                int k = -1;
                for(int kk=0; kk<3; kk++) if (kk != i and kk != j) k = kk;
                std::array<int, 3> order = {i, j, k};
                if (i == 0 and j == 1 and k == 2) continue; // no transpose, no need to check

                std::vector<box3d> outboxes;
                for(auto b : ordered_outboxes) outboxes.push_back(box3d(b.low, b.high, order));

                #ifdef HAS_CPU_BACKEND
                heffte::plan_options options = default_options<default_cpu_backend>();
                options.use_alltoall = std::is_same<variant, using_alltoall>::value;

                auto mpi_tanspose_shaper  = make_reshape3d<default_cpu_backend>(inboxes, outboxes, comm, options);
                auto mpi_direct_shaper    = make_reshape3d<default_cpu_backend>(inboxes, ordered_outboxes, comm, options);
                auto cpu_transpose_shaper = make_reshape3d<default_cpu_backend>(ordered_outboxes, outboxes, comm, options);

                std::vector<scalar_type> result(outboxes[me].count());
                std::vector<scalar_type> reference(outboxes[me].count());
                std::vector<scalar_type> workspace( // allocate one workspace vector for all reshape operations
                    std::max(std::max(mpi_tanspose_shaper->size_workspace(), mpi_direct_shaper->size_workspace()), cpu_transpose_shaper->size_workspace())
                );

                mpi_direct_shaper->apply(input.data(), result.data(), workspace.data());
                cpu_transpose_shaper->apply(result.data(), reference.data(), workspace.data());

                std::fill(result.begin(), result.end(), 0.0); // clear the temporary
                mpi_tanspose_shaper->apply(input.data(), result.data(), workspace.data());

                tassert(match(result, reference));
                #endif

                #ifdef Heffte_ENABLE_GPU
                heffte::plan_options cuoptions = default_options<gpu_backend>();
                cuoptions.use_alltoall = std::is_same<variant, using_alltoall>::value;

                auto cumpi_tanspose_shaper = make_reshape3d<gpu_backend>(inboxes, outboxes, comm, cuoptions);
                auto cumpi_direct_shaper   = make_reshape3d<gpu_backend>(inboxes, ordered_outboxes, comm, cuoptions);
                auto cuda_transpose_shaper = make_reshape3d<gpu_backend>(ordered_outboxes, outboxes, comm, cuoptions);

                gpu::vector<scalar_type> cuinput = gpu::transfer::load(input);
                gpu::vector<scalar_type> curesult(outboxes[me].count());
                gpu::vector<scalar_type> cureference(outboxes[me].count());
                gpu::vector<scalar_type> cuworkspace( // allocate one workspace vector for all reshape operations
                    std::max(std::max(cumpi_tanspose_shaper->size_workspace(), cumpi_direct_shaper->size_workspace()), cuda_transpose_shaper->size_workspace())
                );

                cumpi_direct_shaper->apply(cuinput.data(), curesult.data(), cuworkspace.data());
                cuda_transpose_shaper->apply(curesult.data(), cureference.data(), cuworkspace.data());

                curesult = gpu::vector<scalar_type>(outboxes[me].count());
                cumpi_tanspose_shaper->apply(cuinput.data(), curesult.data(), cuworkspace.data());

                tassert(match(curesult, gpu::transfer::unload(cureference)));
                #endif
            }
        }
    }

}

void perform_tests_cpu(){
    MPI_Comm const comm = MPI_COMM_WORLD;

    test_boxes(comm);

    #ifdef Heffte_ENABLE_FFTW
    switch(mpi::comm_size(comm)) {
        // note that the number of boxes must match the comm size
        // that is the product of the last three of the box dimensions
        case 4:
            test_cpu<10, 13, 10, 2, 2, 1, float, default_cpu_backend, using_alltoall>(comm);
            test_cpu<10, 20, 17, 2, 2, 1, double, default_cpu_backend, using_alltoall>(comm);
            test_cpu<30, 10, 10, 2, 2, 1, std::complex<float>, default_cpu_backend, using_alltoall>(comm);
            test_cpu<11, 10, 13, 2, 2, 1, std::complex<double>, default_cpu_backend, using_alltoall>(comm);
            test_cpu<10, 13, 10, 2, 2, 1, float, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<10, 20, 17, 2, 2, 1, double, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<30, 10, 10, 2, 2, 1, std::complex<float>, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<11, 10, 13, 2, 2, 1, std::complex<double>, default_cpu_backend, using_pointtopoint>(comm);
            break;
        case 12:
            test_cpu<13, 13, 10, 3, 4, 1, float, default_cpu_backend, using_alltoall>(comm);
            test_cpu<16, 21, 17, 2, 3, 2, double, default_cpu_backend, using_alltoall>(comm);
            test_cpu<38, 13, 20, 1, 4, 3, std::complex<float>, default_cpu_backend, using_alltoall>(comm);
            test_cpu<41, 17, 15, 3, 2, 2, std::complex<double>, default_cpu_backend, using_alltoall>(comm);
            test_cpu<13, 13, 10, 3, 4, 1, float, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<16, 21, 17, 2, 3, 2, double, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<38, 13, 20, 1, 4, 3, std::complex<float>, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<41, 17, 15, 3, 2, 2, std::complex<double>, default_cpu_backend, using_pointtopoint>(comm);
            break;
        default:
            // unknown test
            break;
    }
    #endif


    #ifdef Heffte_ENABLE_MKL
    switch(mpi::comm_size(comm)) {
        // note that the number of boxes must match the comm size
        // that is the product of the last three of the box dimensions
        case 4:
            test_cpu<10, 13, 10, 2, 2, 1, float, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<10, 20, 17, 2, 2, 1, double, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<30, 10, 10, 2, 2, 1, std::complex<float>, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<11, 10, 13, 2, 2, 1, std::complex<double>, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<10, 13, 10, 2, 2, 1, float, heffte::backend::mkl, using_pointtopoint>(comm);
            test_cpu<10, 20, 17, 2, 2, 1, double, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<30, 10, 10, 2, 2, 1, std::complex<float>, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<11, 10, 13, 2, 2, 1, std::complex<double>, default_cpu_backend, using_pointtopoint>(comm);
            break;
        case 12:
            test_cpu<13, 13, 10, 3, 4, 1, float, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<16, 21, 17, 2, 3, 2, double, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<38, 13, 20, 1, 4, 3, std::complex<float>, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<41, 17, 15, 3, 2, 2, std::complex<double>, heffte::backend::mkl, using_alltoall>(comm);
            test_cpu<13, 13, 10, 3, 4, 1, float, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<16, 21, 17, 2, 3, 2, double, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<38, 13, 20, 1, 4, 3, std::complex<float>, default_cpu_backend, using_pointtopoint>(comm);
            test_cpu<41, 17, 15, 3, 2, 2, std::complex<double>, default_cpu_backend, using_pointtopoint>(comm);
            break;
        default:
            // unknown test
            break;
    }
    #endif
}

void perform_tests_gpu(){
    #if defined(Heffte_ENABLE_CUDA) or defined(Heffte_ENABLE_ROCM)
    MPI_Comm const comm = MPI_COMM_WORLD;

    switch(mpi::comm_size(comm)) {
        // note that the number of boxes must match the comm size
        // that is the product of the last three of the box dimensions
        case 4:
            test_gpu<10, 13, 10, 2, 2, 1, float, gpu_backend, using_alltoall>(comm);
            test_gpu<10, 20, 17, 2, 2, 1, double, gpu_backend, using_alltoall>(comm);
            test_gpu<30, 10, 10, 2, 2, 1, std::complex<float>, gpu_backend, using_alltoall>(comm);
            test_gpu<11, 10, 13, 2, 2, 1, std::complex<double>, gpu_backend, using_alltoall>(comm);
            test_gpu<10, 13, 10, 2, 2, 1, float, gpu_backend, using_pointtopoint>(comm);
            test_gpu<10, 20, 17, 2, 2, 1, double, gpu_backend, using_pointtopoint>(comm);
            test_gpu<30, 10, 10, 2, 2, 1, std::complex<float>, gpu_backend, using_pointtopoint>(comm);
            test_gpu<11, 10, 13, 2, 2, 1, std::complex<double>, gpu_backend, using_pointtopoint>(comm);
            break;
        case 12:
            test_gpu<13, 13, 10, 3, 4, 1, float, gpu_backend, using_alltoall>(comm);
            test_gpu<16, 21, 17, 2, 3, 2, double, gpu_backend, using_alltoall>(comm);
            test_gpu<38, 13, 20, 1, 4, 3, std::complex<float>, gpu_backend, using_alltoall>(comm);
            test_gpu<41, 17, 15, 3, 2, 2, std::complex<double>, gpu_backend, using_alltoall>(comm);
            test_gpu<13, 13, 10, 3, 4, 1, float, gpu_backend, using_pointtopoint>(comm);
            test_gpu<16, 21, 17, 2, 3, 2, double, gpu_backend, using_pointtopoint>(comm);
            test_gpu<38, 13, 20, 1, 4, 3, std::complex<float>, gpu_backend, using_pointtopoint>(comm);
            test_gpu<41, 17, 15, 3, 2, 2, std::complex<double>, gpu_backend, using_pointtopoint>(comm);
            break;
        default:
            // unknown test
            break;
    }
    #endif
}

void perform_tests_reorder(){
    MPI_Comm const comm = MPI_COMM_WORLD;

    if (mpi::comm_size(comm) == 4){
        test_direct_reordered<double>(comm);
        test_direct_reordered<std::complex<float>>(comm);
        test_reshape_transposed<float, using_alltoall>(comm);
        test_reshape_transposed<std::complex<double>, using_alltoall>(comm);
        test_reshape_transposed<double, using_pointtopoint>(comm);
        test_reshape_transposed<std::complex<float>, using_pointtopoint>(comm);
    }
}

void perform_all_tests(){
    all_tests<> name("heffte reshape methods");
    perform_tests_cpu();
    perform_tests_gpu();
    perform_tests_reorder();
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_all_tests();

    MPI_Finalize();

    return 0;
}
