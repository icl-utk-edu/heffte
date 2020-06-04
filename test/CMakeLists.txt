macro(heffte_add_mpi_test)
    cmake_parse_arguments(_heffte "" "NAME;COMMAND;RANKS" "" ${ARGN} )
    add_test(${_heffte_NAME} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_heffte_RANKS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${_heffte_COMMAND} ${MPIEXEC_POSTFLAGS})
    unset(_heffte_NAME)
    unset(_heffte_RANKS)
    unset(_heffte_COMMAND)
endmacro()

# tests the consistency of the headers, it is a compile time test
add_executable(test_heffte_header  test_heffte_header.cpp)
target_link_libraries(test_heffte_header  Heffte)

# no MPI test of different aspects of the on-node algorithms
add_executable(test_unit_nompi test_units_nompi.cpp)
target_link_libraries(test_unit_nompi  Heffte)
add_test(unit_tests_nompi  test_unit_nompi)

# test the reshape methods
add_executable(test_reshape3d test_reshape3d.cpp)
target_link_libraries(test_reshape3d  Heffte)
heffte_add_mpi_test(NAME heffte_reshape3d_np4  COMMAND test_reshape3d RANKS 4)
heffte_add_mpi_test(NAME heffte_reshape3d_np7  COMMAND test_reshape3d RANKS 7)
heffte_add_mpi_test(NAME heffte_reshape3d_np12 COMMAND test_reshape3d RANKS 12)

# test the complex-to-complex class
add_executable(test_fft3d  test_fft3d.h test_fft3d.cpp)
target_link_libraries(test_fft3d  Heffte)
heffte_add_mpi_test(NAME heffte_fft3d_np2   COMMAND test_fft3d RANKS 2)
heffte_add_mpi_test(NAME heffte_fft3d_np6   COMMAND test_fft3d RANKS 6)
heffte_add_mpi_test(NAME heffte_fft3d_np8   COMMAND test_fft3d RANKS 8)
heffte_add_mpi_test(NAME heffte_fft3d_np12  COMMAND test_fft3d RANKS 12)

# test the real-to-complex class
add_executable(test_fft3d_r2c  test_fft3d.h test_fft3d_r2c.cpp)
target_link_libraries(test_fft3d_r2c  Heffte)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np2   COMMAND test_fft3d_r2c RANKS 2)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np6   COMMAND test_fft3d_r2c RANKS 6)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np8   COMMAND test_fft3d_r2c RANKS 8)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np12  COMMAND test_fft3d_r2c RANKS 12)

if (Heffte_ENABLE_TRACING)
    add_executable(test_trace  test_trace.cpp)
    target_link_libraries(test_trace  Heffte)
    heffte_add_mpi_test(NAME test_tracing   COMMAND test_trace RANKS 2)
endif()

################################################################
# OLD CODE for the OLD API
################################################################
if (Heffte_ENABLE_FFTW)
    #
    # CPU executable double precision
    #
    # source files list
    # include dirs
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    set (test3d_cpu_src test3d_cpu.cpp)
    set (test3d_cpu_r2c_src test3d_cpu_r2c.cpp)

    add_executable(test3d_cpu ${test3d_cpu_src})
    add_executable(test3d_cpu_r2c ${test3d_cpu_src})

    target_link_libraries(test3d_cpu      Heffte)
    target_link_libraries(test3d_cpu_r2c  Heffte)
endif()

if(BUILD_GPU)
  #
  # GPU executable double precision
  #
  set(test3d_gpu_src test3d_gpu.cpp)
  set(test3d_gpu_r2c_src test3d_gpu_r2c.cpp)

  cuda_add_executable(test3d_gpu ${test3d_gpu_src})
  cuda_add_executable(test3d_gpu_r2c ${test3d_gpu_r2c_src})

  set_target_properties(test3d_gpu     PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu heffte_gpu ${CUDA_LIBRARIES})

  set_target_properties(test3d_gpu_r2c PROPERTIES COMPILE_FLAGS "-DFFT_CUFFT")
  target_link_libraries(test3d_gpu_r2c heffte_gpu ${CUDA_LIBRARIES})
endif(BUILD_GPU)
