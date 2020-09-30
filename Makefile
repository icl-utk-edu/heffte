#
# Edit the general compiler options and the backends sections
# but only if the corresponding variables cannot be set
# in the environment or if the backend will actually be used,
# e.g., no need to set CUDA and NVCC variables if cufft is not used
#

############################################################
# General compiler options
############################################################
backends ?= fftw

MPI_ROOT ?= /usr
MPICXX ?= $(MPI_ROOT)/bin/mpicxx
MPICXX_FLAGS ?= -O3 -std=c++11 -I./include/
SHARED_FLAG ?= -fPIC
MPIRUN ?= $(MPI_ROOT)/bin/mpirun
MPIRUN_NUMPROC_FLAG ?= -n
MPIRUN_PREFLAGS ?= --host localhost:12

# Used only by cuda backends
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc
NVCC_FLAGS ?= -std=c++11
NVCC_FINAL_FLAGS ?= -I./include/ -I$(MPI_ROOT)/include/ $(NVCC_FLAGS)

# library, change to ./lib/libheffte.a to build the static libs
libsuffix = .so


############################################################
# Backends, only needs the ones selected for build
############################################################
FFTW_ROOT ?= /usr
FFTW_INCLUDES = -I$(FFTW_ROOT)/include/
FFTW_LIBRARIES = -L$(FFTW_ROOT)/lib/ -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -pthread

CUFFT_INCLUDES = -I$(CUDA_ROOT)/include/
CUFFT_LIBRARIES = -L$(CUDA_ROOT)/lib64/ -lcufft -lcudart

MKL_ROOT ?= /opt/mkl
MKL_INCLUDES = -I$(MKL_ROOT)/include/
MKL_LIBRARIES = -L$(MKL_ROOT)/lib/ -lmkl_cdft_core -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lm -ldl -lgfortran -liomp5


############################################################
# Processing variables
############################################################
comma := ,
space :=
space +=
spaced_backends := $(subst $(comma),$(space),$(backends))
# spaced_backends converts "fftw,cufft" into "fftw cufft"

# variables to be
INCS =
LIBS =
CONFIG_BACKEND = config_fftw
CUDA_KERNELS =

# non-cuda object files
OBJECT_FILES = heffte_fft3d.o     \
               heffte_fft3d_r2c.o \
               heffte_reshape3d.o \
               heffte_plan_logic.o\

# check each backend and set the dependencies
CUFFTW = no_cuda
ifneq (,$(filter cufft,$(spaced_backends)))
	CUFFTW = with_cuda
	INCS += $(CUFFT_INCLUDES)
	LIBS += $(CUFFT_LIBRARIES)
	CUDA_KERNELS += kernels.obj
endif

MKL = no_mkl
ifneq (,$(filter mkl,$(spaced_backends)))
	MKL = with_mkl
	INCS += $(MKL_INCLUDES)
	LIBS += $(MKL_LIBRARIES)
endif

FFTW = no_fftw
ifneq (,$(filter fftw,$(spaced_backends)))
	FFTW = with_fftw
	INCS += $(FFTW_INCLUDES)
	LIBS += $(FFTW_LIBRARIES)
endif

libheffte = ./lib/libheffte$(libsuffix)
ifneq (,$(filter .so,$(libsuffix)))
	MPICXX_FLAGS += $(SHARED_FLAG)
	NVCC_FINAL_FLAGS += -Xcompiler $(SHARED_FLAG)
endif


############################################################
# build rules
############################################################
.PHONY.: all
all: $(libheffte) test_reshape3d test_units_nompi test_fft3d test_fft3d_r2c speed3d_c2c speed3d_r2c

.PHONY.: help
help:
	@echo ""
	@echo "HeFFTe GNU Make Build system, examples:"
	@echo ""
	@echo "    make backends=fftw   FFTW_ROOT=/opt/fftw-3.3.8"
	@echo "    make backends=cufft  CUDA_ROOT=/usr/local/cuda MPI_ROOT=/opt/openmpi-cuda/"
	@echo "    make backends=fftw,cufft  MKL_ROOT=/opt/mkl  libheffte=./lib/libheffte.a"
	@echo "    make ctest"
	@echo "    make clean"
	@echo ""
	@echo "Options set in the environment, the command line, or by editing the Makefile"
	@echo "  backends : comma separated list of backends to include in the library"
	@echo "             fftw,cufft,mkl"
	@echo " libsuffix : specifies whether to build shared of static library"
	@echo "             use .so (default) or .a"
	@echo "  MPI_ROOT : path to the MPI installation, default /usr"
	@echo " CUDA_ROOT : path to the CUDA installation, default /usr/local/cuda"
	@echo " FFTW_ROOT : path to the FFTW3 installation, default /usr"
	@echo "  MKL_ROOT : path to the MKL installation, default /opt/mkl"
	@echo ""
	@echo "Note: the build system assumes that the usual install conventions are followed,"
	@echo "      e.g., using <root>/bin and <root>/include and <root>/lib"
	@echo "      But, if those fail or if additional compiler options and flags are needed,"
	@echo "      those can be adjusted by editing the top two sections of the Makefile."
	@echo "      Alternatively, use the more robust CMake script."
	@echo ""

./include/heffte_config.h:
	cp ./include/heffte_config.cmake.h ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_MAJOR@|1|g' ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_MINOR@|0|g' ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_PATCH@|1|g' ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_TRACING|#undef Heffte_ENABLE_TRACING|g' ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_DISABLE_GPU_AWARE_MPI|#undef Heffte_DISABLE_GPU_AWARE_MPI|g' ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_MAGMA|#undef Heffte_ENABLE_MAGMA|g' ./include/heffte_config.h

.PHONY.: with_fftw no_fftw with_cufft no_cufft
# set heffte_config.h with and without fftw
with_fftw: ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_FFTW|#define Heffte_ENABLE_FFTW|g' ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_ROCM|#undef Heffte_ENABLE_ROCM|g' ./include/heffte_config.h

no_fftw: ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_FFTW|#undef Heffte_ENABLE_FFTW|g' ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_ROCM|#undef Heffte_ENABLE_ROCM|g' ./include/heffte_config.h

# set heffte_config.h with and without cufft
with_mkl: ./include/heffte_config.h $(FFTW)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_MKL|#define Heffte_ENABLE_MKL|g' ./include/heffte_config.h

no_mkl: ./include/heffte_config.h $(FFTW)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_MKL|#undef Heffte_ENABLE_MKL|g' ./include/heffte_config.h

# set heffte_config.h with and without cufft
with_cuda: ./include/heffte_config.h $(MKL)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_CUDA|#define Heffte_ENABLE_CUDA|g' ./include/heffte_config.h

no_cuda: ./include/heffte_config.h $(MKL)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_CUDA|#undef Heffte_ENABLE_CUDA|g' ./include/heffte_config.h

# cuda object files
kernels.obj: $(CUFFTW)
	$(NVCC) $(NVCC_FINAL_FLAGS) -c ./src/heffte_backend_cuda.cu -o kernels.obj

# build the object files
%.o: src/%.cpp $(CUFFTW) $(CUDA_KERNELS)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -c $< -o $@

# library targets
./lib/libheffte.so: $(OBJECT_FILES)
	mkdir -p lib
	$(MPICXX) -shared $(OBJECT_FILES) $(CUDA_KERNELS) -o ./lib/libheffte.so $(LIBS)

./lib/libheffte.a: $(OBJECT_FILES)
	mkdir -p lib
	ar rcs ./lib/libheffte.a $(OBJECT_FILES) $(CUDA_KERNELS)


############################################################
# building tests
############################################################
test_reshape3d: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -L./lib/ ./test/test_reshape3d.cpp -o test_reshape3d $(libheffte) $(LIBS)

test_units_nompi: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -L./lib/ ./test/test_units_nompi.cpp -o test_units_nompi $(libheffte) $(LIBS)

test_fft3d: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -L./lib/ ./test/test_fft3d.cpp -o test_fft3d $(libheffte) $(LIBS)

test_fft3d_r2c: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -L./lib/ ./test/test_fft3d_r2c.cpp -o test_fft3d_r2c $(libheffte) $(LIBS)

speed3d_c2c: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -I./benchmarks/ -L./lib/ ./benchmarks/speed3d_c2c.cpp -o speed3d_c2c $(libheffte) $(LIBS)

speed3d_r2c: $(libheffte)
	$(MPICXX) $(MPICXX_FLAGS) $(INCS) -I./test/ -I./benchmarks/ -L./lib/ ./benchmarks/speed3d_r2c.cpp -o speed3d_r2c $(libheffte) $(LIBS)

# execute the tests
.PHONY.: ctest
ctest:
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  4 $(MPIRUN_PREFLAGS) ./test_reshape3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  7 $(MPIRUN_PREFLAGS) ./test_reshape3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG) 12 $(MPIRUN_PREFLAGS) ./test_reshape3d
	./test_units_nompi
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  1 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  2 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  4 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  6 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  8 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG) 12 $(MPIRUN_PREFLAGS) ./test_fft3d
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  1 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  2 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  4 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  6 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG)  8 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c
	$(MPIRUN) $(MPIRUN_NUMPROC_FLAG) 12 $(MPIRUN_PREFLAGS) ./test_fft3d_r2c


############################################################
# clean
############################################################
.PHONY.: clean
clean:
	rm -fr ./include/heffte_config.h
	rm -fr *.o
	rm -fr *.obj
	rm -fr lib
	rm -fr test_reshape3d
	rm -fr test_units_nompi
	rm -fr test_fft3d_r2c
	rm -fr test_fft3d
	rm -fr speed3d_c2c
	rm -fr speed3d_r2c
