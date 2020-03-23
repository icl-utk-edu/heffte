/*
    -- heFFTe (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       Performance test for 3D FFTs using heFFTe
*/

#include "test_fft3d.h"
#include <iostream>


void perform_tests(MPI_Comm const comm){
    all_tests<> name("heffte::fft class");
    int const num_ranks = mpi::comm_size(comm);
    test_fft3d_arrays<backend::fftw, float, 19, 20, 21>(comm);
}

int main(int argc, char *argv[]){
    
    int me, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm fft_comm = MPI_COMM_WORLD;  // Change if need to compute FFT within a subcommunicator
    MPI_Comm_rank(fft_comm, &me);

    // Create a C2C plan

    // Allocate, inplace fft overwrites input

    // Warming up runnings

    // Initialize data as random numbers

    // Execute FFT

    // Validate result

    // Print results and timing

    // Free memory

    MPI_Finalize();
    return 0;
}
