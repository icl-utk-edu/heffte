#-- HEFFTE (version 0.2) --
# Univ. of Tennessee, Knoxville
# Standard makefile for linux systems.

SHELL 			= /bin/sh

# Tracing flags
MYFLAGS			=
#MYFLAGS	  = -DTRACING_HEFFTE

# ----------------------------------------

# Compiler - linker section
CC          =	mpicxx $(MYFLAGS)  #-fopenmp
CCFLAGS     =	-g -O3 -std=c++11 #-pg
SHFLAGS     =	-fPIC
ARCHIVE     = ar
ARCHFLAGS   = -rc
SHLIBFLAGS  = -shared

# CUDA compilation
GPU_TARGET ?=   Volta Kepler Pascal
NVCC       ?=   nvcc
NVCCFLAGS  ?=   -O3 -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function" -std=c++11

# -----------------------------------------------------------------------------------

ifneq ($(findstring Kepler, $(GPU_TARGET)),)
    GPU_TARGET += sm_30 sm_35
endif
ifneq ($(findstring Maxwell, $(GPU_TARGET)),)
    GPU_TARGET += sm_50
endif
ifneq ($(findstring Pascal, $(GPU_TARGET)),)
    GPU_TARGET += sm_60
endif
ifneq ($(findstring Volta, $(GPU_TARGET)),)
    GPU_TARGET += sm_70
endif
ifneq ($(findstring Turing, $(GPU_TARGET)),)
    GPU_TARGET += sm_75
endif

# -----------------------------------------------------------------------------------

# GPU architecture flags

NV_SM    :=
NV_COMP  :=

ifneq ($(findstring sm_10, $(GPU_TARGET)),)
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm_13, $(GPU_TARGET)),)
    $(warning CUDA arch 1.x is no longer supported by CUDA >= 6.x and MAGMA >= 2.0)
endif
ifneq ($(findstring sm_20, $(GPU_TARGET)),)
    MIN_ARCH ?= 200
    NV_SM    += -gencode arch=compute_20,code=sm_20
    NV_COMP  := -gencode arch=compute_20,code=compute_20
    $(warning CUDA arch 2.x is no longer supported by CUDA >= 9.x)
endif
ifneq ($(findstring sm_30, $(GPU_TARGET)),)
    MIN_ARCH ?= 300
    NV_SM    += -gencode arch=compute_30,code=sm_30
    NV_COMP  := -gencode arch=compute_30,code=compute_30
endif
ifneq ($(findstring sm_32, $(GPU_TARGET)),)
    MIN_ARCH ?= 320
    NV_SM    += -gencode arch=compute_32,code=sm_32
    NV_COMP  := -gencode arch=compute_32,code=compute_32
endif
ifneq ($(findstring sm_35, $(GPU_TARGET)),)
    MIN_ARCH ?= 350
    NV_SM    += -gencode arch=compute_35,code=sm_35
    NV_COMP  := -gencode arch=compute_35,code=compute_35
endif
ifneq ($(findstring sm_50, $(GPU_TARGET)),)
    MIN_ARCH ?= 500
    NV_SM    += -gencode arch=compute_50,code=sm_50
    NV_COMP  := -gencode arch=compute_50,code=compute_50
endif
ifneq ($(findstring sm_52, $(GPU_TARGET)),)
    MIN_ARCH ?= 520
    NV_SM    += -gencode arch=compute_52,code=sm_52
    NV_COMP  := -gencode arch=compute_52,code=compute_52
endif
ifneq ($(findstring sm_53, $(GPU_TARGET)),)
    MIN_ARCH ?= 530
    NV_SM    += -gencode arch=compute_53,code=sm_53
    NV_COMP  := -gencode arch=compute_53,code=compute_53
endif
ifneq ($(findstring sm_60, $(GPU_TARGET)),)
    MIN_ARCH ?= 600
    NV_SM    += -gencode arch=compute_60,code=sm_60
    NV_COMP  := -gencode arch=compute_60,code=compute_60
endif
ifneq ($(findstring sm_61, $(GPU_TARGET)),)
    MIN_ARCH ?= 610
    NV_SM    += -gencode arch=compute_61,code=sm_61
    NV_COMP  := -gencode arch=compute_61,code=compute_61
endif
ifneq ($(findstring sm_62, $(GPU_TARGET)),)
    MIN_ARCH ?= 620
    NV_SM    += -gencode arch=compute_62,code=sm_62
    NV_COMP  := -gencode arch=compute_62,code=compute_62
endif
ifneq ($(findstring sm_70, $(GPU_TARGET)),)
    MIN_ARCH ?= 700
    NV_SM    += -gencode arch=compute_70,code=sm_70
    NV_COMP  := -gencode arch=compute_70,code=compute_70
endif
ifneq ($(findstring sm_71, $(GPU_TARGET)),)
    MIN_ARCH ?= 710
    NV_SM    += -gencode arch=compute_71,code=sm_71
    NV_COMP  := -gencode arch=compute_71,code=compute_71
endif
ifneq ($(findstring sm_75, $(GPU_TARGET)),)
    MIN_ARCH ?= 750
    NV_SM    += -gencode arch=compute_75,code=sm_75
    NV_COMP  := -gencode arch=compute_75,code=compute_75
endif
ifeq ($(NV_COMP),)
    $(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of Fermi, Kepler, Maxwell, Pas\
cal, Volta, Turing, or valid sm_[0-9][0-9]. Please edit your make.inc file)
endif
NVCCFLAGS += $(NV_SM) $(NV_COMP)

# -------------------------------------------------------------------------------
# 1d FFT library
# fft     = CUFFT
fft     		= FFTW3
FFT     		= $(shell echo $(fft) | tr a-z A-Z)
FFT_INC 		= -I../include -DFFT_$(FFT)

ifneq ($(timing),)
	FFT_INC  += -DHEFFTE_TIME_DETAILED
endif



ifneq (,$(filter CUFFTW CUFFT CUFFT_M CUFFT_R,$(FFT)))
FFT_INC		 += -I$(CUDADIR)/include/  -L$(CUDADIR)/lib64/ -lcufft -lcudart -lnvToolsExt
endif

# # -------------------------------------------------------------------------------
# # 1DFFT library
# Provide fftw3 folders or (if available) use "module load fftw3"
FFT1D_DIR 	= /home/aayalaob/fftw-3.3.8
FFT1D_INC 	= -I$(FFT1D_DIR)/api/
FFT1D_LIB 	= -L$(FFT1D_DIR)/lib/

# -------------------------------------------------------------------------------
# Memory aligment (if needed)
MEM_INC   	=	-DFFT_MEMALIGN=64

# -------------------------------------------------------------------------------
# Library build, no edit needed after this line.
LIB3D      	= libheffte.a
SHLIB3D    	= libheffte.so
SRC3D_CPU   = heffte_common.cpp heffte_fft3d.cpp heffte_reshape3d.cpp heffte_trace.cpp heffte.cpp heffte_wrap.cpp
ifneq (,$(filter CUFFTW CUFFT CUFFT_M CUFFT_R,$(FFT)))
SRC3D_GPU   = heffte_pack3d.cu heffte_scale.cu
else
SRC3D_CPU  += heffte_pack3d.cpp
endif

INC3D       = heffte_common.h heffte_fft3d.h heffte_reshape3d.h heffte_trace.h heffte.h heffte_utils.h heffte_pack3d.h heffte_wrap.h
OBJ3D_CPU   = $(SRC3D_CPU:.cpp=.o)
OBJ3D_GPU   = $(SRC3D_GPU:.cu=.o)
OBJ3D 			= $(OBJ3D_CPU) $(OBJ3D_GPU)

OBJ_CPU   	= $(sort $(OBJ3D_CPU))
OBJ_GPU   	= $(OBJ3D_GPU)

# -------------------------------------------------------------------------------
# tBuilding argets
lib	 : lib3d
shlib: shlib3d
all	 : shlib lib

help:
	@echo 'make                default = lib, KISS 1d fft, double precision'
	@echo 'make lib            build static hefft libs: 3d'
	@echo 'make shlib          build shared hefft libs: 3d'
	@echo 'make all            build 4 hefft libs: lib and shlib'
	@echo 'make lib3d          build static 3d hefft lib
	@echo 'make shlib3d        build shared 3d hefft lib
	@echo 'make ... fft=fftw   build with FFTW3 (FFTW), FFTW2, MKL 1d ffts'
	@echo 'make ... p=single   build for single precision'
	@echo 'make clean          remove all *.o files'
	@echo 'make clean-all      remove *.o and lib files'
	@echo 'make tar            create a tarball, 2 levels up'

lib3d:
	$(MAKE) clean
	$(MAKE) static3d fft=$(fft)

shlib3d:
	$(MAKE) clean
	$(MAKE) shared3d fft=$(fft)

static3d:	$(OBJ3D)
		$(ARCHIVE) $(ARCHFLAGS) $(LIB3D) $(OBJ3D)

shared3d:	$(OBJ3D)
		$(CC) $(CCFLAGS) $(SHFLAGS) $(SHLIBFLAGS) -o $(SHLIB3D) $(OBJ3D)

clean:
	@rm -f *.o *.pyc

cleanall:
	@rm -f *.o *.pyc lib*.a lib*.so

tar:
	cd ../..; tar cvf cslib.tar cslib/README cslib/LICENSE \
		cslib/doc cslib/src cslib/test

# rules
heffte_fft3d.o:heffte_fft3d.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
heffte_reshape3d.o:heffte_reshape3d.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
heffte_common.o:heffte_common.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
heffte.o:heffte.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
heffte_trace.o:heffte_trace.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
heffte_wrap.o:heffte_wrap.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<

# Rules for CUDA compilation
ifneq (,$(filter CUFFTW CUFFT CUFFT_M CUFFT_R,$(FFT)))
%.o:%.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(FFT_INC) $(FFT1D_INC)  -c $< -o $@
else
heffte_pack3d.o:heffte_pack3d.cpp
	$(CC) $(CCFLAGS) $(SHFLAGS) $(MEM_INC) $(FFT_INC) $(FFT1D_INC) -c $<
endif

echo:
	@echo "OBJ3D     = $(OBJ3D)"
	@echo "SRC3D_CPU = $(SRC3D_CPU)"
	@echo "SRC3D_GPU = $(SRC3D_GPU)"
	@echo "OBJ_CPU 	 = $(OBJ_CPU)"
	@echo "SRC_CPU 	 = $(OBJ_CPU:.o=.cpp)"
	@echo "FFT       = $(FFT)"
	@echo "FFT_INC   = $(FFT_INC)"
