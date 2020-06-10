HEFFTE Flags
============
We provide two application programming interfaces (APIs). The new API (from version 1.0), is based on C++11. Benchmarks are available in the *benchmarks* folder. The best choice of options are tunned for each backend, and can also be manually defined by users via the following flags:

- `-reorder`: reorder the elements of the arrays so that each 1-D FFT will use contiguous data.

- `-no-reorder`: some of the 1-D will be strided (non contiguous).

- `-a2a`: use MPI_Alltoallv() communication method.

- `-p2p`: use MPI_Send() and MPI_Irecv() communication methods.

- `-pencils`: use pencil reshape logic.

- `-slabs`: use slab reshape logic.

- `-io_pencils`: if input and output proc grids are pencils, useful for comparison with other libraries.

- `-mps`: for the CUFFT backend and multiple GPUs, it associates the mpi ranks with different cuda devices, using CudaSetDevice(my_rank%device_count). It is deactivated by default.

The old API (from version 0.2), was based on standard C++98, and examples to test it are available within the folder *tests/*, flags that user can define are listed below:

- `-c`: communication  can be set to flag (default = point):
          point = point-to-point comm
          all = use MPI_all2all collective

- `-e`: exchange flag can be set to (default = pencil):
          pencil = pencil to pencil data exchange (4 stages for full FFT)
          brick = brick to pencil data exchange (6 stages for full FFT)

- `-p`: select pack/unpack for methods of data reshaping, can be set to (default = memcpy):
          array = array based
          ptr = pointer based
          memcpy = memcpy based

- `-s`: scales FFT after forward computation

- `-t`: print split timing of routines (default = deactivated)

- `-r`: call HEFFTE to compute only data reshapes without computation (default = deactivated)

- `-o`: print full array before/after computation (default = deactivated)

- `-m` : FFT computation requirement for the test, can be set to (default = 0):
          0 = computes a forward and backward FFT.
          1 = computes only a forward FFT.

- `-v`: Check correctness of FFT computation (default = deactivated)

- `-verb`: Verbosity flag, prints hardware and extra information (default = deactivated)
