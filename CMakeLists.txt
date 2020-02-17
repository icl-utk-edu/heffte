cmake_minimum_required(VERSION 3.10)

project("heffte" VERSION 0.2.0 LANGUAGES CXX)

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
#set(OpenMP_CXX_FLAGS "-lpthread")
# set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${OpenMP_CXX_FLAGS} " )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3" )


# #########
# CPU BUILD
# #########
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

if(HEFFTE_TIME_DETAILED)
  add_definitions(-DHEFFTE_TIME_DETAILED)
endif(HEFFTE_TIME_DETAILED)


# build CPU libheffte
add_library(heffte ${libheffte_SRCS})
target_link_libraries(heffte HEFFTE::FFTW MPI::MPI_CXX)

# install
install(TARGETS heffte EXPORT HEFFTE_Targets DESTINATION lib)
install(EXPORT HEFFTE_Targets FILE HEFFTE_Targets.cmake NAMESPACE HEFFTE:: DESTINATION lib/cmake/HEFFTE)
include(CMakePackageConfigHelpers)
write_basic_package_version_file("HEFFTEConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/HEFFTEConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/HEFFTEConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HEFFTEConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/HEFFTEConfigVersion.cmake" DESTINATION lib/cmake/HEFFTE)
#install(DIRECTORY heffte DESTINATION include FILES_MATCHING PATTERN "include/*.h")
install (DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h"
  PATTERN "**~" EXCLUDE
  PATTERN "*_gpu*" EXCLUDE
  PATTERN "*.cu*" EXCLUDE
  PATTERN "*_cuda*" EXCLUDE
  )
target_include_directories(heffte PUBLIC $<INSTALL_INTERFACE:include>)


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
  cuda_add_library(heffte_gpu ${libheffte_gpu_SRCS} OPTIONS "-DFFT_CUFFT")

  # the following is necessary so that .cpp sources files passed to
  # cuda_add_library get the right compile flags
  set_target_properties(heffte_gpu PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")

  target_link_libraries(heffte_gpu MPI::MPI_CXX)

  # install
  install(TARGETS heffte_gpu EXPORT HEFFTE_Targets DESTINATION lib)

endif(BUILD_GPU)

####################
# EXAMPLES / TESTS #
####################
add_subdirectory(test)
