/*
    -- heFFTe (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       Performance test for 3D FFTs using heFFTe
*/

#include "test_fft3d.h"

using namespace heffte;

template <typename backend_tag> 
void benchmark_gpu_tester(std::array<int,3>& N){

    double t = 0.0, t_max = 0.0; // timing parameters
    int me, nprocs;
    MPI_Comm fft_comm = MPI_COMM_WORLD;  // Change if need to compute FFT within a subcommunicator
    MPI_Comm_rank(fft_comm, &me);
    MPI_Comm_size(fft_comm, &nprocs);

    // std::array<int, 3> N = {64,64,64};

    // Get grid of processors at input and output
    std::array<int,3> proc_i = {0,0,0}; 
    std::array<int,3> proc_o = {0,0,0}; 
    heffte_proc_setup(N.data(), proc_i.data(), nprocs);
    heffte_proc_setup(N.data(), proc_o.data(), nprocs);

    // Create input and output boxes on local processor
    box3d const world = {{0, 0, 0}, {N[0]-1, N[1]-1, N[2]-1}};
    std::vector<box3d> inboxes  = heffte::split_world(world, proc_i);
    std::vector<box3d> outboxes = heffte::split_world(world, proc_o);

    // Define 3D FFT plan
    heffte::fft3d<backend_tag> fft(inboxes[me], outboxes[me], fft_comm);
    
    // Locally initialize input
    auto input = make_data<std::complex<float>>(inboxes[me]);

    // Define output arrays on host
    std::vector<std::complex<float>> output(fft.size_outbox()); 
    std::vector<std::complex<float>> inverse(fft.size_inbox());
   
    // copying arrays to GPU
    auto input_gpu   = cuda::load(input); 
    auto output_gpu  = cuda::load(output);
    auto inverse_gpu = cuda::load(inverse);

    // mpi::dump(0, input, "Initial FFT data");

    // Warmup
    fft.forward(input.data(), output.data(),  scale::full);
    fft.backward(output.data(), inverse.data());

    // Execution    
    int ntest = 1; 
    t -= MPI_Wtime();
    for(int i = 0; i < ntest; ++i) {
        fft.forward(input.data(), output.data(),  scale::full);
        fft.backward(output.data(), inverse.data());
    }
    t += MPI_Wtime();

    // Get execution time
	MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, fft_comm);    

    // Copy back to the host
    inverse = cuda::unload(inverse_gpu);
    // mpi::dump(0, inverse, "After backward FFT");

    // Validate result
    tassert(approx(input, inverse)); 
    
    // Print results
    if(me==0){
        t_max = t_max / (2.0 * ntest);
        double fftsize  = 1.0 * N[0] * N[1] * N[2];
        double floprate = 5.0 * fftsize * log(fftsize) * 1e-9 / log(2.0) / t_max;
        cout << "------------------------------- \n";
        cout << "heFFTe performance test on GPUs \n";
        cout << "------------------------------- \n";
        cout << "Backend: " << backend::name<backend_tag>() << endl;
        cout << "Size: " << N[0] << "x" << N[1] << "x" << N[2] << endl;
        cout << "Nprc: " << nprocs << endl;
        cout << "Time: " << t_max << " (s)" << endl;
        cout << "Perf: " << floprate << " GFlops/s" << endl;
        cout << "Tolr: " << precision<std::complex<float>>::tolerance << endl;
    }
}


int main(int argc, char *argv[]){

    if (argc < 4){
        cout << "Usage: mpirun -np x ./speed3d nx ny nz \n ";
        cout << "       where nx, ny, nz are the 3D array dimensions \n";
        return 0;
    }

    MPI_Init(&argc, &argv);
    std::array<int,3> size_fft = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}; // FFT size from user

    benchmark_gpu_tester<backend::cufft>( size_fft );

    MPI_Finalize();
    return 0;
}
