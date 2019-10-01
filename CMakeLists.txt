# require cmake 2.8.0 for findCUDA.cmake
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0 FATAL_ERROR)

##################
project("heffte")
# The version number.
set (heffte_VERSION_MAJOR 0)
set (heffte_VERSION_MINOR 1)
##################

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# Set default install path t o build
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

# allow cmake to use custom modules (e.g. FindFFTW)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


find_package(FFTW REQUIRED)
find_package(MPI REQUIRED)
# find_package(OpenMP REQUIRED)
# Make the include path visible to all targets
INCLUDE_DIRECTORIES (
  ${heffte_SOURCE_DIR}/include
  ${heffte_BINARY_DIR}/include
  )



# Add OpenMP and FFTW cxx flags to all targets
# FOR PGI COMPILER
#set(OpenMP_C_FLAGS "-lpthread")
#set(OpenMP_CXX_FLAGS "-lpthread")
# set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  ${OpenMP_C_FLAGS} -I${FFTW_INCLUDES}" )
set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -I${FFTW_INCLUDES}" )
# set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS} -I${FFTW_INCLUDES}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -I${FFTW_INCLUDES}" )

# some default behavior
if(BUILD_SHARED)
  set(BUILD_SHARED_LIBS ON)
else()
  set(BUILD_SHARED_LIBS OFF)
endif(BUILD_SHARED)

# #########
# CPU BUILD
# #########
include(FindMPI)

# set cmake compiler to mpi compiler wrapper found by FindMPI (usually mpicxx)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )

#include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# libheffte source files list
set(libheffte_SRCS
  src/heffte_common.cpp
  src/heffte.cpp
  src/heffte_fft3d.cpp
  src/heffte_pack3d.cpp
  src/heffte_reshape3d.cpp
  src/heffte_trace.cpp
  )

add_definitions(-DFFT_FFTW3)

# build CPU libheffte (shared or static ?)
if(BUILD_SHARED_LIBS)
  add_library(heffte SHARED ${libheffte_SRCS})
else(BUILD_SHARED_LIBS)
  add_library(heffte STATIC ${libheffte_SRCS})
endif(BUILD_SHARED_LIBS)
target_link_libraries(heffte ${FFTW_LIB} ${FFTWF_LIB})


# install
install(TARGETS heffte DESTINATION lib)
#install(DIRECTORY heffte DESTINATION include FILES_MATCHING PATTERN "include/*.h")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h"
  PATTERN "**~" EXCLUDE
  PATTERN "*_gpu*" EXCLUDE
  PATTERN "*.cu*" EXCLUDE
  PATTERN "*_cuda*" EXCLUDE
  )


# ##########
# GPU BUILD
# ##########
if(BUILD_GPU)
  remove_definitions(-DFFT_FFTW3)
  find_package(CUDA REQUIRED)

  if(NOT CUDA_NVCC_FLAGS)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
    list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)
  endif()

  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo)

  # libheffte_gpu source files list
  set(libheffte_gpu_SRCS
    src/heffte.cpp
    src/heffte_common.cpp
    src/heffte_fft3d.cpp
    src/heffte_pack3d.cu
    src/heffte_scale.cu
    src/heffte_reshape3d.cpp
    src/heffte_trace.cpp
    )


  # GPU libheffte_gpu
  if(BUILD_SHARED_LIBS)
    cuda_add_library(heffte_gpu SHARED ${libheffte_gpu_SRCS} OPTIONS "-DFFT_CUFFT_A")
  else(BUILD_SHARED_LIBS)
    cuda_add_library(heffte_gpu STATIC ${libheffte_gpu_SRCS} OPTIONS "-DFFT_CUFFT_A")
  endif(BUILD_SHARED_LIBS)

  # the following is necessary so that .cpp sources files passed to
  # cuda_add_library get the right compile flags
  set_target_properties(heffte_gpu PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT_A")

  # install
  install(TARGETS heffte_gpu DESTINATION lib)
  install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h")

endif(BUILD_GPU)

####################
# EXAMPLES / TESTS #
####################
add_subdirectory(test)