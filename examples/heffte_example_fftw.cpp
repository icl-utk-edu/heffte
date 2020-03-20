#include "heffte.h"

/*!
 * \brief HeFFTe example 1, simple DFT using two MPI ranks and FFTW backend.
 *
 * Performing DFT on three dimensional data in a box of 4 by 4 by 4 split
 * across the third dimension between two MPI ranks.
 */
void compute_dft(MPI_Comm comm){

    int me; // this process rank within the comm
    MPI_Comm_rank(comm, &me);

    int num_ranks; // total number of ranks in the comm
    MPI_Comm_size(comm, &num_ranks);

    if (num_ranks != 2){
        if (me == 0) std::cout << " heffte_example_fftw is set to 2 mpi ranks, exiting \n";
        return;
    }

    // define the domain split between two boxes
    heffte::box3d const left_box  = {{0, 0, 0}, {3, 3, 1}};
    heffte::box3d const right_box = {{0, 0, 2}, {3, 3, 3}};

    // the box associated with this MPI rank
    heffte::box3d const my_box = (me == 0) ? left_box : right_box;

    // define the heffte class and the input and output geometry
    heffte::fft3d<heffte::backend::fftw> fft(my_box, my_box, comm);

    // vectors with the correct sizes to store the input and output data
    // taking the size of the input and output boxes
    std::vector<std::complex<double>> input(fft.size_inbox());
    std::vector<std::complex<double>> output(fft.size_outbox());

    // fill the input vector with data that looks like 0, 1, 2, ...
    std::iota(input.begin(), input.end(), 0); // put some data in the input

    // perform a forward DFT
    fft.forward(input.data(), output.data());

    // check the accuracy
    if (me == 1){
        // given the data, the solution on MPI rank 1 is very simple
        std::vector<std::complex<double>> reference_output(fft.size_outbox());
        reference_output[0] = {-512.0, 0.0};

        // compare the computed to the actual and throw error if there is a mismatch
        for(int i=0; i<fft.size_outbox(); i++)
            if (std::abs(reference_output[i] - output[i]) > 1.E-14)
                throw std::runtime_error("discrepancy between the reference and actual output");
    }

    // reset the input to zero
    std::fill(input.begin(), input.end(), std::complex<double>(0.0, 0.0));

    // perform a backward DFT
    fft.backward(output.data(), input.data());

    // rescale the result
    for(auto &i : input) i /= 64.0;

    // compare the computed entries to the original input data
    double err = 0.0;
    for(int i=0; i<fft.size_inbox(); i++)
        err = std::max(err, std::abs(input[i] - std::complex<double>(double(i), 0.0)));

    // print the error for each MPI rank
    std::cout << std::scientific;
    if (me == 0) std::cout << "rank 0 computed error: " << err << std::endl;
    MPI_Barrier(comm);
    if (me == 1) std::cout << "rank 1 computed error: " << err << std::endl;
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_dft(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
