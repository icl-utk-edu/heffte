
#include "heffte.h"
#include "test_fft3d.h"

/*
 * This method should be EMPTY in every pull request.
 * The goal is to have a file that is part of the build system
 * but it is not en encumbered by template parameters or complex text logic.
 * A single test for just one backend or one case of options/inputs
 * can be easily isolated here and tested.
 * This can also be used for profiling and running benchmarks
 * on sub-modules, e.g., a single reshape operation
 * or direct call to a backend.
 */
void test_sandbox(MPI_Comm const comm){
    // add code here to be tested by the system

    int me = mpi::comm_rank(comm);
    int nprocs = mpi::comm_size(comm);
    box3d<int> const world = {{0, 0, 0}, {40, 40, 40}};

    // Get grid of processors at input and output
    std::array<int,3> proc_i = heffte::proc_setup_min_surface(world, nprocs);

    std::vector<box3d<int>> inboxes  = heffte::split_world(world, proc_i);
    std::vector<box3d<int>> outboxes  = heffte::split_world(world, proc_i);

    int batch_size = 10;

    std::vector<double> input(batch_size * inboxes[me].count(), 0);
    std::vector<double> output(batch_size * inboxes[me].count(), 0);

    std::iota(input.begin(), input.end(), 0);


    std::unique_ptr<reshape3d_alltoall<tag::cpu, direct_packer, int>> reshape =
                make_reshape3d_alltoall<tag::cpu, direct_packer, int>(nullptr, inboxes, outboxes, false, comm);


    std::vector<double> workspace(batch_size * reshape->size_workspace());
    reshape->apply(batch_size, input.data(), output.data(), workspace.data());

    mpi::dump(0, std::vector<double>(input.begin(), input.begin() + 10), "input");
    mpi::dump(0, std::vector<double>(output.begin(), output.begin() + 10), "output");

    tassert(approx(input, output));

}

void test_sandbox(){
    // same as above, but no MPI will be used
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    test_sandbox();
    test_sandbox(MPI_COMM_WORLD);

    MPI_Finalize();
}
