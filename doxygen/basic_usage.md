# Basic Usage

HeFFTe provides two interfaces, a new one based on C++11 features and succinct templates, and an older one using C++98 features and is based on the [fftMPI](https://fftmpi.sandia.gov/) library but allow for the cuFFT backend. The C++98 interface is provided for compatibility purposes and will be removed in the near future, new projects should use the C++11 interface.

There is no significant performance difference between the two interfaces, since they both implement the same fundamental algorithms. However, the new interface uses RAII style of resource management and is therefore more stable with respect to memory errors and is easier to maintain. Both classes uses the same distributed index logic, but the new API is more flexible in terms of index ordering and generally makes fewer assumptions on the inputs. The new API also supports multiple backends in a single binary and is both const correct and type-safe.

### New C++11 API

The interface and all associated backends are included in the `libheffte` library which can be linked against using the CMake target `Heffte::Heffte`. The `heffte.h` header includes all needed definitions and the HeFFTe methods are wrapped in the `heffte::` namespace. The user describes a distributed FFT problem with either a `heffte::fft3d` or a `heffte::fft3d_r2c` objects. The global set of indexes is distributed across the ranks of an MPI communicators where each rank has an input and output sub-box of the indexes, those are described with `box3d` structures that list the lowest and highest index in each direction. In addition to the distribution of the indexes described by the boxes and the associated MPI communicator, the the constructors of the fft classes accept a `plan_options` struct set of options about using algorithm details, e.g., point-to-point or all-to-all communication algorithm, strided vs contiguous 1d transforms, and others.

The actual transforms are computed with the `forward()` and `backward()` methods that provide overloads that work with either vector or vector-like containers or raw-arrays, and optional external work-buffers. The result can also be scaled, per user request. A single class can handle transforms with different precision and data-types, so long as the distribution of the indexes is the same. The C++11 API also provides type safety and compatibility with `std::complex` types as well as the types provided by the backends, e.g., `fftw_complex`.

See also the examples in `<install-prefix>/share/heffte/examples/` or `<source-prefix>/examples/`.

### Old C++98 API

The interface can support only one backend per binary library, hence there are `libheffte` and `libheffte_gpu` libraries corresponding the CPU (FFTW3 or MKL) backend and the GPU (CUDA) backend. There are also corresponding CMake targets `Heffte::Heffte` and `Heffte::heffte_gpu`. The same header is used as in the C++11 API, i.e., `heffte.h` but the namespace is capitalized `HEFFTE`. The transforms are performed by the `HEFFTE::FFT3d` class with constructor that takes only the MPI communicator. A separate `setup()` or `setup_r2c()` method call is needed to describe the distribution of the indexes and to form the transform logic. The class templates on `float` or `double` precision, arrays of complex numbers have to be cast to `float/double`, and the input arrays have to be padded. Overall the interface is less flexible and therefore deprecated.

