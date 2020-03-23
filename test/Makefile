#-- HEFFTE (version 0.2) --
# Univ. of Tennessee, Knoxville
# Standard makefile for linux systems.

SHELL = /bin/sh

# Tracing flags
MYFLAGS		 =
#MYFLAGS	  = -DTRACING_HEFFTE


# ----------------------------------------

MPICC     = mpicxx $(MYFLAGS)  #-pg # -fopenmp
MPIF90    = mpif90
# MPIF90    = mpif90  -g -fcheck=all -Wall
CCFLAGS   =	-g -O3 -std=c++11
F90FLAGS  =	-g -O3 -cpp
LINKFLAGS =	-g

# ----------------------------------------
MEM_INC =	-DFFT_MEMALIGN=64
# ----------------------------------------

# 1d FFT library
# fft     = CUFFT
fft     = FFTW3
FFT     = $(shell echo $(fft) | tr a-z A-Z)

# -------------------------------------------------------------------------------
# 1DFFT library
# Provide fftw3 folders or (if available) use "module load fftw3"
ifneq (,$(filter FFTW3,$(FFT)))
FFT1D_DIR 	= /home/aayalaob/fftw-3.3.8
FFT1D_INC 	= -I$(FFT1D_DIR)/api/
FFT1D_LIB 	= -L$(FFT1D_DIR)/lib/ -lfftw3 -lfftw3f
endif

ifneq (,$(filter MKL,$(FFT)))
FFT1D_DIR 	= /spack/opt/spack/linux-scientific7-x86_64/gcc-7.3.0/intel-mkl-2019.3.199-2pn4ijrdu2xxlbbp45o5jqq2c2vyh6kj/compilers_and_libraries_2019.3.199/linux/mkl
FFT1D_INC  += -I$(FFT1D_DIR)/include/
FFT1D_LIB 	= -L$(FFT1D_DIR)/lib/ -lmkl_cdft_core -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lm -ldl -lgfortran -liomp5
endif


# ----------------------------------------
# Building test executables, no edit needed after this line.

ifneq (,$(filter CUFFT CUFFT_M CUFFT_R, $(FFT)))
    FFT1D_LIB = -L$(CUDADIR)/lib64/ -lcufft -lcudart -lnvToolsExt
endif

FFT_INC    =   -I../include -DFFT_$(FFT)
FFT3D_LIB  =   -L../src

ifneq (,$(filter CUFFTW CUFFT CUFFT_M CUFFT_R,$(FFT)))
FFT_INC		+=   -I$(CUDADIR)/include/
endif

ifneq ($(timing),)
	FFT_INC  += -DHEFFTE_TIME_DETAILED
endif

# -----------------------------------------------------------------------------------
# targets

all: cleanall	test3d_cpu_r2c test3d_cpu test3d_gpu_r2c test3d_gpu
tests_cpu:  clean_cpu test3d_cpu_r2c test3d_cpu test3d_cpu_fortran
tests_gpu:  clean_gpu test3d_gpu_r2c test3d_gpu

test3d_cpu_r2c:	test3d_cpu_r2c.o
	$(MPICC) $(LINKFLAGS) $(FFT3D_LIB) test3d_cpu_r2c.o -lheffte $(FFT1D_LIB) -o test3d_cpu_r2c

test3d_cpu:	test3d_cpu.o
	$(MPICC) $(LINKFLAGS) $(FFT3D_LIB) test3d_cpu.o -lheffte $(FFT1D_LIB) -o test3d_cpu

test3d_gpu_r2c:	test3d_gpu_r2c.o
	$(MPICC) $(LINKFLAGS) $(FFT3D_LIB) test3d_gpu_r2c.o -lheffte $(FFT1D_LIB) -o test3d_gpu_r2c

test3d_gpu:	test3d_gpu.o
	$(MPICC) $(LINKFLAGS) $(FFT3D_LIB) test3d_gpu.o -lheffte $(FFT1D_LIB) -o test3d_gpu

test3d_fortran:	heffte.o test3d_fortran.o
	$(MPIF90) $(LINKFLAGS) $(FFT3D_LIB) test3d_fortran.o heffte.o -lheffte -lstdc++ $(FFT1D_LIB) -o test3d_fortran

clean:
	@rm -f *.o *.mod

clean_cpu:
	@rm -f *.o *.mod test3d_cpu test3d_cpu

clean_gpu:
	@rm -f *.o *.mod test3d_gpu test3d_gpu

cleanall:
	@rm -f *.o *.mod test3d_gpu test3d_gpu_r2c test3d_cpu test3d_cpu_r2c test3d_fortran *.a

# rules

%.o:%.cpp
	$(MPICC) $(CCFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<

%.o:%.f90
	$(MPIF90) $(F90FLAGS) $(MEM_INC) $(FFT_INC)  $(FFT1D_INC) -c $<
