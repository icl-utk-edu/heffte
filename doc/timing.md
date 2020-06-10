Timing and profiling HEFFTE
===========================

We provide tests and benchmarks to perform a wide variety of experiments. In particular, we provide users with benchmarks to evaluate performance and to conduct comparison with other FFT libraries. For our old API, the following performance tests are available (c.f., folder *tests/*):
~~~
test3d_cpu, test3d_gpu
~~~

For the new interface (release v1.0), our benchmarks are (c.f., folder *benchmarks/*):

~~~
speed3d_c2c, speed3d_r2c
~~~

These executables are available after compilation, and instructions are provided within files. When using the compilation flag -DHEFFTE_TIME_DETAILED=true, a double timing array is provided, where runtimes for different tasks of the algorithm can be obtained as:

timing_array[0] = total computation for of FFT

timing_array[1] = perform low dimensional FFTs (CUFFT, FFTW3, MKL)

timing_array[2] = packing data

timing_array[3] = unpacking data

timing_array[4] = FFT scaling

timing_array[5] = all-to-all communication

timing_array[6] = total MPI communication

To report performance for the computation of FFTs, we recommend performing a warmup call before initializing the input data, and then time the corresponding FFT execution kernel using a loop, 5 to 10 iterations are recommended, and then report the average.

We have developed a tool to get a detailed execution trace useful for visualization, by compiling with the flag:

~~~
cmake  -DHeffte_ENABLE_TRACING=ON ...
~~~

User can also link to sophisticate visualization software, such as *Vampir*. Simply load the Vampir module on your machine, and add the following flags to the CMake compilation:

~~~
SCOREP_WRAPPER=off SCOREP_WRAPPER_ARGS=--cuda cmake -DCMAKE_CXX_COMPILER="scorep-mpicxx" -DHeffte_ENABLE_CUDA=ON ... 
~~~

Visualization can provide useful insights on how kernels are performing and compare CPU versus GPU versions, as in the following figure, where we show a profile obtained with Vampir using 32 nodes and a 3D FFT of size 1024x1024x1024.

![heffte_kernels](https://bitbucket.org/aayala32/logos/raw/be7a2ac8c7c0d70083db2f1c109afa71224f63e8/heffte_kernels.png)


[img_latex]: # (\image html "../figures/heffte_kernels.png" width=10cm)
[img_html]: # (\image latex "../figures/heffte_kernels.png" width=10cm)