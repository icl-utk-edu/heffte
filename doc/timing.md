Timing HEFFTE
=============

The *tester codes* allow to calculate the *runtime* of HEFFTE functions. A double timer-array of size 8 is provided, where the timing for different tasks of the algorithm is saved as:

timing_array[0] = total computation for of FFT

timing_array[1] = perform low dimensional FFTs (CUFFT, FFTW3, etc)

timing_array[2] = packing data

timing_array[3] = unpacking data

timing_array[4] = FFT scaling

timing_array[5] = all-to-all communication

timing_array[6] = total MPI communication

The total execution time can be obtained as follows:

double time_fft = 0;
time_fft -= MPI_Wtime();
      heffte_execute(in, out, FORWARD);
time_fft +-= MPI_Wtime();


It is recommended to perform one or two warmup runs by calling the corresponding *execute* function, before timing and profiling. Refer to the *test* folder for examples.
