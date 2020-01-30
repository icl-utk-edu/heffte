/*
    -- heFFTe (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
       Testing R2C Fast Fourier Transform on distributed CPUs
       @author Alan Ayala
*/

#include "testing.h"

using namespace HEFFTE_NS;

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing R2C 3D FFT
*/
int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  MPI_Comm fft_comm = MPI_COMM_WORLD;  // Change if need to compute FFT within a subcommunicator

  int me, nprocs;
  MPI_Comm_size(fft_comm, &nprocs);
  MPI_Comm_rank(fft_comm, &me);

  heffte_init();

  // Select your type of data, input and output
  float *work_in, *work_out;
  // double *work_in, *work_out;

  // Create fft object according to your data type
  FFT3d<float> *fft = new FFT3d<float>(fft_comm);
  // FFT3d <double> *fft = new FFT3d<double>(fft_comm);

  fft->mem_type = HEFFTE_MEM_CPU;  // setting internal memory type

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

  // Create R2C plan
  opts.timeplan -= MPI_Wtime();
    heffte_plan_r2c_create(work_in, fft, opts.N, i_lo, i_hi, o_lo, o_hi, opts.workspace);
  opts.timeplan += MPI_Wtime();

  MPI_Barrier(fft_comm);
  opts.timeinit += MPI_Wtime();  // End initialization timing

  // Allocate input and output arrays, FFT is always out-of-place for R2C case
  heffte_allocate(HEFFTE_MEM_CPU, &work_in,  opts.workspace[0], opts.nbytes);     // input 3D-real array
  heffte_allocate(HEFFTE_MEM_CPU, &work_out, opts.workspace[0], opts.nbytes);    // output 3D-complex array


  // Warming up runnings
  heffte_execute_r2c(fft, work_in, work_out);  // R2C FFT computation


  // Initialize data as random real numbers
  heffte_initialize_host(work_in, nfft_in, opts.seed, HEFFTE_REAL_DATA);
  if (opts.oflag) opts.heffte_print_grid(0, "Input data", work_in, nfft_in, i_lo, i_hi, o_lo, o_hi);

  // Set FFT timing vector to zero
  memset(timing_array, 0, NTIMING_VARIABLES * sizeof(double));

  // heffte_tracing_init(); // To obtain traces you must compile sources and test defining -DTRACING_HEFFTE
  opts.timefft -= MPI_Wtime();
    for (int i = 0; i < opts.nloop; i++)
      heffte_execute_r2c(fft, work_in, work_out);  // R2C FFT computation
  opts.timefft += MPI_Wtime();
  // heffte_tracing_finalize();

  if (opts.oflag) opts.heffte_print_grid(1, "Computed R2C FFT", work_out, nfft_in, i_lo, i_hi, o_lo, o_hi);

  // Print results and timing
  opts.heffte_timing(fft);

  // Free memory
  delete fft;
  heffte_deallocate(HEFFTE_MEM_CPU, work_in);
  heffte_deallocate(HEFFTE_MEM_CPU, work_out);
  MPI_Finalize();

}
