# include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# CPU executable double precision
#
# source files list
set (test3d_cpu_src test3d_cpu.cpp)
set (test3d_cpu_r2c_src test3d_cpu_r2c.cpp)

add_definitions(-DFFT_FFTW3)

add_executable(test3d_cpu ${test3d_cpu_src})
add_executable(test3d_cpu_r2c ${test3d_cpu_src})

# set_target_properties(test3d_cpu PROPERTIES COMPILE_FLAGS "-DFFT_FFTW3")
target_link_libraries(test3d_cpu     heffte)
target_link_libraries(test3d_cpu_r2c heffte)


if(BUILD_GPU)
  #
  # GPU executable double precision
  #
  set(test3d_gpu_src test3d_gpu.cpp)
  set(test3d_gpu_r2c_src test3d_gpu_r2c.cpp)

  remove_definitions(-DFFT_FFTW3)

  cuda_add_executable(test3d_gpu ${test3d_gpu_src})
  cuda_add_executable(test3d_gpu_r2c ${test3d_gpu_r2c_src})

  set_target_properties(test3d_gpu     PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu heffte_gpu ${CUDA_LIBRARIES})

  set_target_properties(test3d_gpu_r2c PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu_r2c heffte_gpu ${CUDA_LIBRARIES})

  cuda_add_cufft_to_target(test3d_gpu)
  cuda_add_cufft_to_target(test3d_gpu_r2c)

endif(BUILD_GPU)
