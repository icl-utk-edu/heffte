cmake_minimum_required(VERSION 3.10)

project("heffte" VERSION 0.2.0 LANGUAGES CXX)

if (HEFFTE_ENABLE_DOXYGEN)
    add_subdirectory(doxygen)
endif()

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
  src/heffte_wrap.cpp
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

  set( GPU_TARGET "Kepler Maxwell Pascal" CACHE STRING "CUDA architectures to compile for; one or more of Fermi, Kepler, Maxwell, Pascal, Volta, or valid sm_[0-9][0-9]" )

  if (CUDA_FOUND)
      message( STATUS "Found CUDA ${CUDA_VERSION}" )
      message( STATUS "    CUDA_INCLUDE_DIRS:   ${CUDA_INCLUDE_DIRS}"   )
      message( STATUS "    CUDA_CUDART_LIBRARY: ${CUDA_CUDART_LIBRARY}" )
      include_directories( ${CUDA_INCLUDE_DIRS} )

      # NVCC options for the different cards
      # sm_xx is binary, compute_xx is PTX for forward compatability
      # MIN_ARCH is lowest requested version
      # NV_SM    accumulates sm_xx for all requested versions
      # NV_COMP  is compute_xx for highest requested version
      set( NV_SM    "" )
      set( NV_COMP  "" )

      set(CUDA_SEPARABLE_COMPILATION ON)

      if (${GPU_TARGET} MATCHES Fermi)
          set( GPU_TARGET "${GPU_TARGET} sm_20" )
      endif()

      if (${GPU_TARGET} MATCHES Kepler)
          set( GPU_TARGET "${GPU_TARGET} sm_30 sm_35" )
      endif()

      if (${GPU_TARGET} MATCHES Maxwell)
          set( GPU_TARGET "${GPU_TARGET} sm_50" )
      endif()

      if (${GPU_TARGET} MATCHES Pascal)
          set( GPU_TARGET "${GPU_TARGET} sm_60" )
      endif()

      if (${GPU_TARGET} MATCHES Volta)
          set( GPU_TARGET "${GPU_TARGET} sm_70" )
      endif()

      if (${GPU_TARGET} MATCHES Turing)
          set( GPU_TARGET "${GPU_TARGET} sm_75" )
      endif()

      if (${GPU_TARGET} MATCHES sm_20)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 200 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_20,code=sm_20 )
          set( NV_COMP        -gencode arch=compute_20,code=compute_20 )
          message( STATUS "    compile for CUDA arch 2.0 (Fermi)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_30)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 300 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_30,code=sm_30 )
          set( NV_COMP        -gencode arch=compute_30,code=compute_30 )
          message( STATUS "    compile for CUDA arch 3.0 (Kepler)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_35)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 300 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_35,code=sm_35 )
          set( NV_COMP        -gencode arch=compute_35,code=compute_35 )
          message( STATUS "    compile for CUDA arch 3.5 (Kepler)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_50)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 500 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_50,code=sm_50 )
          set( NV_COMP        -gencode arch=compute_50,code=compute_50 )
          message( STATUS "    compile for CUDA arch 5.0 (Maxwell)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_52)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 520 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_52,code=sm_52 )
          set( NV_COMP        -gencode arch=compute_52,code=compute_52 )
          message( STATUS "    compile for CUDA arch 5.2 (Maxwell)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_53)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 530 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_53,code=sm_53 )
          set( NV_COMP        -gencode arch=compute_53,code=compute_53 )
          message( STATUS "    compile for CUDA arch 5.3 (Maxwell)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_60)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 600 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_60,code=sm_60 )
          set( NV_COMP        -gencode arch=compute_60,code=compute_60 )
          message( STATUS "    compile for CUDA arch 6.0 (Pascal)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_61)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 610 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_61,code=sm_61 )
          set( NV_COMP        -gencode arch=compute_61,code=compute_61 )
          message( STATUS "    compile for CUDA arch 6.1 (Pascal)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_62)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 620 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_62,code=sm_62 )
          set( NV_COMP        -gencode arch=compute_62,code=compute_62 )
          message( STATUS "    compile for CUDA arch 6.2 (Pascal)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_70)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 700 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_70,code=sm_70 )
          set( NV_COMP        -gencode arch=compute_70,code=compute_70 )
          message( STATUS "    compile for CUDA arch 7.0 (Volta)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_71)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 710 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_71,code=sm_71 )
          set( NV_COMP        -gencode arch=compute_71,code=compute_71 )
          message( STATUS "    compile for CUDA arch 7.1 (Volta)" )
      endif()

      if (${GPU_TARGET} MATCHES sm_75)
          if (NOT MIN_ARCH)
              set( MIN_ARCH 750 )
          endif()
          set( NV_SM ${NV_SM} -gencode arch=compute_75,code=sm_75 )
          set( NV_COMP        -gencode arch=compute_75,code=compute_75 )
          message( STATUS "    compile for CUDA arch 7.5 (Turing)" )
      endif()

      if (NOT MIN_ARCH)
          message( FATAL_ERROR "GPU_TARGET must contain one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing, or valid sm_[0-9][0-9]" )
      endif()

          set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -DHAVE_CUBLAS ${NV_SM} ${NV_COMP} ${FORTRAN_CONVENTION} )
      add_definitions( "-DHAVE_CUBLAS -DMIN_CUDA_ARCH=${MIN_ARCH}" )
      message( STATUS "Define -DHAVE_CUBLAS -DMIN_CUDA_ARCH=${MIN_ARCH}" )
  endif()


  # libheffte_gpu source files list
  set(libheffte_gpu_SRCS
    src/heffte.cpp
    src/heffte_common.cpp
    src/heffte_fft3d.cpp
    src/heffte_pack3d.cu
    src/heffte_scale.cu
    src/heffte_reshape3d.cpp
    src/heffte_trace.cpp
    src/heffte_wrap.cpp
    )


  # GPU libheffte_gpu
  cuda_add_library(heffte_gpu ${libheffte_gpu_SRCS} OPTIONS "-DFFT_CUFFT")

  # the following is necessary so that .cpp sources files passed to
  # cuda_add_library get the right compile flags
  set_target_properties(heffte_gpu PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_include_directories(heffte_gpu PUBLIC $<INSTALL_INTERFACE:include>)

  target_link_libraries(heffte_gpu MPI::MPI_CXX)
  cuda_add_cufft_to_target(heffte_gpu)

  # install
  install(TARGETS heffte_gpu EXPORT HEFFTE_Targets DESTINATION lib)

endif(BUILD_GPU)

####################
# EXAMPLES / TESTS #
####################
add_subdirectory(test)

enable_testing()
add_test(heffte_test_cpu ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
${CMAKE_CURRENT_BINARY_DIR}/test/test3d_cpu ${MPIEXEC_POSTFLAGS}  -g 64 64 64 -v -i 82783 -c point -verb -s )
add_test(heffte_test_gpu ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
${CMAKE_CURRENT_BINARY_DIR}/test/test3d_gpu ${MPIEXEC_POSTFLAGS}  -g 64 64 64 -v -i 82783 -c point -verb -s )
