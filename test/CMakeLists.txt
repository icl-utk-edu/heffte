# include dirs
include_directories(${FFTW_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# CPU executable double precision
#
# source files list
set (test3d_cpu_src
  test3d_cpu.cpp
)
add_definitions(-DFFT_FFTW3)

add_executable(test3d_cpu ${test3d_cpu_src})
# set_target_properties(test3d_cpu PROPERTIES COMPILE_FLAGS "-DFFT_FFTW3")
target_link_libraries(test3d_cpu
  heffte
  ${FFTW_THREADS_LIB}
  ${FFTW_LIB}
  ${FFTWF_THREADS_LIB}
  ${FFTWF_LIB}
)

if(BUILD_GPU)
  #
  # GPU executable double precision
  #
  set(test3d_gpu_src
    test3d_gpu.cpp
    )

  remove_definitions(-DFFT_FFTW3)

  cuda_add_executable(test3d_gpu ${test3d_gpu_src})
  set_target_properties(test3d_gpu PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT_A")
  target_link_libraries(test3d_gpu
    heffte_gpu
    ${CUDA_LIBRARIES}
  )

  cuda_add_cufft_to_target(test3d_gpu)

endif(BUILD_GPU)
