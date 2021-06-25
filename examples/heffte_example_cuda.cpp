#include "heffte.h"

/*!
 * \brief HeFFTe example 5, using the cuFFT backend.
 *
 * This example is near identical to the first (fftw) example,
 * the main difference is the use of the cufft backend.
 * The interface and types for the cufft backend work the same,
 * with the exception that the array must sit on the GPU device
 * and if the optional vector interface is used then the vector
 * containers are of type heffte::cuda::vector.
 */
void compute_dft(MPI_Comm comm){

    // wrapper around MPI_Comm_rank() and MPI_Comm_size(), using this is optional
    int const my_rank   = heffte::mpi::comm_rank(comm);
    int const num_ranks = heffte::mpi::comm_size(comm);

    if (num_ranks != 2){
        if (my_rank == 0) std::cout << " heffte_example_cuda is set to 2 mpi ranks, exiting \n";
        return;
    }

    // define the domain split between two boxes
    heffte::box3d<> const left_box  = {{0, 0, 0}, {3, 3, 1}};
    heffte::box3d<> const right_box = {{0, 0, 2}, {3, 3, 3}};

    // the box associated with this MPI rank
    heffte::box3d<> const my_box = (my_rank == 0) ? left_box : right_box;

    if (heffte::gpu::device_count() > 1){
        // on a multi-gpu system, distribute the devices across the mpi ranks
        heffte::gpu::device_set(heffte::mpi::comm_rank(comm) % heffte::gpu::device_count());
    }

    // define the heffte class and the input and output geometry
    // heffte::plan_options can be specified just as in the backend::fftw
    heffte::fft3d<heffte::backend::cufft> fft(my_box, my_box, comm);

    // create some input on the CPU
    std::vector<std::complex<double>> input(fft.size_inbox());
    std::iota(input.begin(), input.end(), 0); // put some data in the input

    // load the input into the GPU memory
    // this is equivalent to cudaMalloc() followed by cudaMemcpy()
    // the destructor of heffte::gpu::vector will call cudaFree()
    heffte::gpu::vector<std::complex<double>> gpu_input = heffte::gpu::transfer().load(input);

    // allocate memory on the device for the output
    heffte::gpu::vector<std::complex<double>> gpu_output(fft.size_outbox());

    // allocate scratch space, this is using the public type alias buffer_container
    // and for the cufft backend this is heffte::gpu::vector
    // for the CPU backends (fftw and mkl) the buffer_container is std::vector
    heffte::fft3d<heffte::backend::cufft>::buffer_container<std::complex<double>> workspace(fft.size_workspace());
    static_assert(std::is_same<decltype(gpu_output), decltype(workspace)>::value,
                  "the containers for the output and workspace have different types");

    // perform forward fft using arrays and the user-created workspace
    fft.forward(gpu_input.data(), gpu_output.data(), workspace.data(), heffte::scale::full);

    // optional step, free the workspace since the inverse will use the vector API
    workspace = heffte::gpu::vector<std::complex<double>>();

    // compute the inverse FFT transform using the container API
    heffte::gpu::vector<std::complex<double>> gpu_inverse = fft.backward(gpu_output);

    // move the result back to the CPU for comparison purposes
    std::vector<std::complex<double>> inverse = heffte::gpu::transfer::unload(gpu_inverse);

    // compute the error between the input and the inverse
    double err = 0.0;
    for(size_t i=0; i<input.size(); i++)
        err = std::max(err, std::abs(inverse[i] - input[i]));

    // print the error for each MPI rank
    std::cout << std::scientific;
    for(int i=0; i<num_ranks; i++){
        MPI_Barrier(comm);
        if (my_rank == i) std::cout << "rank " << i << " computed error: " << err << std::endl;
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_dft(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
