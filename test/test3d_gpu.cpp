/*
    -- heFFTe (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       Testing inplace C2C Fast Fourier Transform on distributed GPUs
       @author Alan Ayala
*/

#include "testing.h"

using namespace HEFFTE;

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing C2C 3D FFT
*/
int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm fft_comm = MPI_COMM_WORLD;  // Change if need to compute FFT within a subcommunicator

  int me, nprocs;
  MPI_Comm_size(fft_comm, &nprocs);
  MPI_Comm_rank(fft_comm, &me);

  heffte_init();

  // Select your type of data, input and output
  float *work;   // on host
  float *dwork; // on device
  // double *work;   // on host
  // double *dwork; // on device


  // Create fft object according to your data type
  FFT3d<float> *fft = new FFT3d<float>(fft_comm);
  // FFT3d <double> *fft = new FFT3d<double>(fft_comm);

  fft->mem_type = HEFFTE_MEM_GPU;  // setting internal memory type

  // Read from command line
  heffte_opts opts(fft_comm);
  opts.parse_opts(argc, argv, fft);


  // Start initialization time
  MPI_Barrier(fft_comm);
  opts.timeinit -= MPI_Wtime();

  // Set up a grid of processors, required if there are not predefined grid of processors
  heffte_proc_setup(opts.N, opts.proc_i, nprocs);
  heffte_proc_setup(opts.N, opts.proc_o, nprocs);


  // Get bricks of data on local processor, required if arrays are not already distributed on a processors grid
  int i_lo[3], i_hi[3];   // local brick vertices at intitial partition
  int o_lo[3], o_hi[3];   // local brick vertices at final partition
  heffte_int_t nfft_in;   // local brick size at initial partition
  heffte_int_t nfft_out;  // local brick size at final partition


  heffte_grid_setup(opts.N, i_lo, i_hi, o_lo, o_hi,
                    opts.proc_i, opts.proc_o, me, nfft_in, nfft_out);

  // Create C2C plan
  opts.timeplan -= MPI_Wtime();
    heffte_plan_create(fft, opts.N, i_lo, i_hi, o_lo, o_hi, opts.permute, opts.workspace);
  opts.timeplan += MPI_Wtime();

  MPI_Barrier(fft_comm);
  opts.timeinit += MPI_Wtime();  // End initialization timing


  // Allocate input and output arrays, FFT is always out-of-place for C2C case
  heffte_allocate(HEFFTE_MEM_CPU, &work,  opts.workspace[0], opts.nbytes);     // input/output 3D array, on host
  heffte_allocate(HEFFTE_MEM_GPU, &dwork, opts.workspace[0], opts.nbytes); // input/output 3D array, on device

  // Warming up runnings
  if (opts.mode == 0) {
    for (int i = 0; i < opts.nloop; i++) {
      heffte_execute(fft, dwork, dwork, FORWARD);
      heffte_execute(fft, dwork, dwork, BACKWARD);
    }
  } else if (opts.mode == 1) {
      for (int i = 0; i < opts.nloop; i++)
        heffte_execute(fft, dwork, dwork, FORWARD);
  }

  // Initialize data as random numbers
  heffte_initialize_host(work, nfft_in, opts.seed, HEFFTE_COMPLEX_DATA);
  if (opts.oflag) opts.heffte_print_grid(0, "Input data", work, nfft_in, i_lo, i_hi, o_lo, o_hi);

  cudaMemcpy(dwork, work, opts.nbytes, cudaMemcpyHostToDevice);


  // Set FFT timing vector to zero
  memset(timing_array, 0, NTIMING_VARIABLES * sizeof(double));

  // heffte_tracing_init(); // To obtain traces you must compile sources and test defining -DTRACING_HEFFTE
  opts.timefft -= MPI_Wtime();
  if (opts.mode == 0) {
    for (int i = 0; i < opts.nloop; i++) {
      heffte_execute(fft, dwork, dwork, FORWARD);   // Forward C2C FFT computation
      heffte_execute(fft, dwork, dwork, BACKWARD);  // Backward C2C FFT computation
    }
  } else if (opts.mode == 1) {
      for (int i = 0; i < opts.nloop; i++)
        heffte_execute(fft, dwork, dwork, FORWARD);  // Forward C2C FFT computation
  }
  opts.timefft += MPI_Wtime();
  // heffte_tracing_finalize();


  // Copy output back to the CPU
  cudaDeviceSynchronize();
  cudaMemcpy(work, dwork, opts.nbytes, cudaMemcpyDeviceToHost);

  if (opts.oflag) opts.heffte_print_grid(1, "Computed C2C FFT", work, nfft_in, i_lo, i_hi, o_lo, o_hi);
  if (opts.vflag) heffte_validate(work, nfft_in, opts.seed, opts.epsmax, fft_comm);  // Error validation

  // Print results and timing
  opts.heffte_timing(fft);

  // Free memory
  delete fft;
  heffte_deallocate(HEFFTE_MEM_CPU, work);
  heffte_deallocate(HEFFTE_MEM_GPU, dwork);
  MPI_Finalize();

}
