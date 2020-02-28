macro(heffte_add_mpi_test)
    cmake_parse_arguments(_heffte "" "NAME;COMMAND;RANKS" "" ${ARGN} )
    add_test(${_heffte_NAME} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_heffte_RANKS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${_heffte_COMMAND} ${MPIEXEC_POSTFLAGS})
    unset(_heffte_NAME)
    unset(_heffte_RANKS)
    unset(_heffte_COMMAND)
endmacro()

add_executable(test_reshape_cpu test_reshape_cpu.cpp)
target_link_libraries(test_reshape_cpu  Heffte)

heffte_add_mpi_test(NAME heffte_reshape_cpu4  COMMAND test_reshape_cpu RANKS 4)
heffte_add_mpi_test(NAME heffte_reshape_cpu7  COMMAND test_reshape_cpu RANKS 7)
heffte_add_mpi_test(NAME heffte_reshape_cpu12 COMMAND test_reshape_cpu RANKS 12)

add_executable(test_unit_nompi test_units_nompi.cpp)
target_link_libraries(test_unit_nompi  Heffte)

add_test(unit_tests_nompi  test_unit_nompi)

#
# CPU executable double precision
#
# source files list
# include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (test3d_cpu_src test3d_cpu.cpp)
set (test3d_cpu_r2c_src test3d_cpu_r2c.cpp)

#add_definitions(-DFFT_FFTW3)

add_executable(test3d_cpu ${test3d_cpu_src})
add_executable(test3d_cpu_r2c ${test3d_cpu_src})

# set_target_properties(test3d_cpu PROPERTIES COMPILE_FLAGS "-DFFT_FFTW3")
target_link_libraries(test3d_cpu      Heffte)
target_link_libraries(test3d_cpu_r2c  Heffte)

if(BUILD_GPU)
  #
  # GPU executable double precision
  #
  set(test3d_gpu_src test3d_gpu.cpp)
  set(test3d_gpu_r2c_src test3d_gpu_r2c.cpp)

  #remove_definitions(-DFFT_FFTW3)

  cuda_add_executable(test3d_gpu ${test3d_gpu_src})
  cuda_add_executable(test3d_gpu_r2c ${test3d_gpu_r2c_src})

  set_target_properties(test3d_gpu     PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu heffte_gpu ${CUDA_LIBRARIES})

  set_target_properties(test3d_gpu_r2c PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu_r2c heffte_gpu ${CUDA_LIBRARIES})

endif(BUILD_GPU)
