#
# Edit the following lines:
# - edit the general compiler section
# - pick a backend library
# - edit the includes and libraries for the backend
# - use make backend=<chosen-backend>
#

############################################################
# General compiler options
############################################################
backends = fftw

MPICXX = mpicxx
MPICXX_FLAGS = -O3 -fPIC -I./include/
MPIRUN = mpirun

# Used only by cuda backends
NVCC = nvcc
NVCC_FLAGS = -I./include/ -I/usr/include/openmpi/ -Xcompiler -fPIC

# library, change to ./lib/libheffte.a to build the static libs
libheffte = ./lib/libheffte.so


############################################################
# Backends, only needs the one needed
############################################################
FFTW_INCLUDES = -I/usr/include/
FFTW_LIBRARIES = -L/usr/lib/x86_64-linux-gnu/ -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -pthread

CUFFT_INCLUDES = -I/usr/local/cuda/include/
CUFFT_LIBRARIES = -L/usr/local/cuda/lib64/ -lcufft -lcudart


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
OBJECT_FILES = heffte_common.o    \
               heffte.o           \
               heffte_fft3d.o     \
               heffte_fft3d_r2c.o \
               heffte_pack3d.o    \
               heffte_reshape3d.o \
               heffte_trace.o     \
               heffte_wrap.o      \

# check each backend and set the dependencies
CUFFTW = no_cuda
ifneq (,$(filter cufft,$(spaced_backends)))
	CUFFTW = with_cuda
	INCS += $(CUFFT_INCLUDES)
	LIBS += $(CUFFT_LIBRARIES)
	CUDA_KERNELS += kernels.obj
endif

FFTW = no_fftw
ifneq (,$(filter fftw,$(spaced_backends)))
	FFTW = with_fftw
	INCS += $(FFTW_INCLUDES)
	LIBS += $(FFTW_LIBRARIES)
endif


############################################################
# build rules
############################################################
.PHONY.: all
all: $(libheffte) test_reshape3d test_units_nompi test_fft3d test_fft3d_r2c

./include/heffte_config.h:
	cp ./include/heffte_config.cmake.h ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_MAJOR@|0|g' ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_MINOR@|2|g' ./include/heffte_config.h
	sed -i -e 's|@Heffte_VERSION_PATCH@|1|g' ./include/heffte_config.h

.PHONY.: with_fftw no_fftw with_cufft no_cufft
# set heffte_config.h with and without fftw
with_fftw: ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_FFTW|#define Heffte_ENABLE_FFTW|g' ./include/heffte_config.h

no_fftw: ./include/heffte_config.h
	sed -i -e 's|#cmakedefine Heffte_ENABLE_FFTW|#undef Heffte_ENABLE_FFTW|g' ./include/heffte_config.h

# set heffte_config.h with and without cufft
with_cuda: ./include/heffte_config.h $(FFTW)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_CUDA|#define Heffte_ENABLE_CUDA|g' ./include/heffte_config.h

no_cuda: ./include/heffte_config.h $(FFTW)
	sed -i -e 's|#cmakedefine Heffte_ENABLE_CUDA|#undef Heffte_ENABLE_CUDA|g' ./include/heffte_config.h

# cuda object files
kernels.obj: $(CUFFTW)
	$(NVCC) $(NVCC_FLAGS) -c ./src/heffte_backend_cuda.cu -o kernels.obj

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


# execute the tests
.PHONY.: ctest
ctest:
	$(MPIRUN) -np  4 test_reshape3d
	$(MPIRUN) -np  7 test_reshape3d
	$(MPIRUN) -np 12 test_reshape3d
	./test_units_nompi
	$(MPIRUN) -np  2 test_fft3d
	$(MPIRUN) -np  6 test_fft3d
	$(MPIRUN) -np  8 test_fft3d
	$(MPIRUN) -np 12 test_fft3d
	$(MPIRUN) -np  2 test_fft3d_r2c
	$(MPIRUN) -np  6 test_fft3d_r2c
	$(MPIRUN) -np  8 test_fft3d_r2c
	$(MPIRUN) -np 12 test_fft3d_r2c


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
