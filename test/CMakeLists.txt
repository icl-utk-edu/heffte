# tests the consistency of the headers, it is a compile time test
add_executable(test_heffte_header  test_heffte_header.cpp)
target_link_libraries(test_heffte_header  Heffte)

# tests the consistency of the template types, it is a compile time test
add_executable(test_heffte_templates  test_heffte_templates.cpp)
target_link_libraries(test_heffte_templates  Heffte)

# sandbox executable for testing purposes
add_executable(sandbox  sandbox.cpp)
target_link_libraries(sandbox  Heffte)

# no MPI test of the formatting helper functions of the complex class
add_executable(test_complex_ostream  test_complex_ostream.cpp)
target_link_libraries(test_complex_ostream  Heffte)
add_test(unit_tests_complex_ostream  test_complex_ostream)
set_tests_properties(unit_tests_complex_ostream PROPERTIES RUN_SERIAL ${Heffte_SEQUENTIAL_TESTING})

# no MPI test of different aspects of the on-node algorithms
add_executable(test_unit_nompi test_units_nompi.cpp)
target_link_libraries(test_unit_nompi  Heffte)
add_test(unit_tests_nompi  test_unit_nompi)
set_tests_properties(unit_tests_nompi PROPERTIES RUN_SERIAL ${Heffte_SEQUENTIAL_TESTING})

# no MPI test of different aspects of the stock implementations
add_executable(test_unit_stock  test_units_stock.cpp)
target_link_libraries(test_unit_stock  Heffte)
add_test(unit_tests_stock  test_unit_stock)
set_tests_properties(unit_tests_stock PROPERTIES RUN_SERIAL ${Heffte_SEQUENTIAL_TESTING})

# test the reshape methods
add_executable(test_reshape3d test_reshape3d.cpp)
target_link_libraries(test_reshape3d  Heffte)
heffte_add_mpi_test(NAME heffte_reshape3d_np4  COMMAND test_reshape3d RANKS 4)
heffte_add_mpi_test(NAME heffte_reshape3d_np7  COMMAND test_reshape3d RANKS 7)
heffte_add_mpi_test(NAME heffte_reshape3d_np12 COMMAND test_reshape3d RANKS 12)

# test the complex-to-complex class
foreach(_heffte_num_procs 1 2 4 6 8 12)
    add_executable(test_fft3d_np${_heffte_num_procs}  test_fft3d.h test_fft3d_np${_heffte_num_procs}.cpp)
    target_link_libraries(test_fft3d_np${_heffte_num_procs}  Heffte)
    heffte_add_mpi_test(NAME heffte_fft3d_np${_heffte_num_procs}  COMMAND  test_fft3d_np${_heffte_num_procs}  RANKS  ${_heffte_num_procs})
endforeach()
unset(_heffte_num_procs)

# test the user-provided streams
add_executable(test_streams  test_fft3d.h test_streams.cpp)
target_link_libraries(test_streams  Heffte)
heffte_add_mpi_test(NAME heffte_streams_np6 COMMAND test_streams RANKS 6) # will enable once Volta issue is fixed

# subcommunicator tests
add_executable(test_subcomm  test_fft3d.h test_subcomm.cpp)
target_link_libraries(test_subcomm  Heffte)
heffte_add_mpi_test(NAME test_subcomm_np8   COMMAND test_subcomm RANKS  8)
heffte_add_mpi_test(NAME test_subcomm_np12  COMMAND test_subcomm RANKS 12)

# test the real-to-complex class
add_executable(test_fft3d_r2c  test_fft3d.h test_fft3d_r2c.cpp)
target_link_libraries(test_fft3d_r2c  Heffte)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np1   COMMAND test_fft3d_r2c RANKS 1)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np2   COMMAND test_fft3d_r2c RANKS 2)
heffte_add_mpi_test(NAME heffte_fft2d_r2c_np4   COMMAND test_fft3d_r2c RANKS 4)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np6   COMMAND test_fft3d_r2c RANKS 6)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np8   COMMAND test_fft3d_r2c RANKS 8)
heffte_add_mpi_test(NAME heffte_fft3d_r2c_np12  COMMAND test_fft3d_r2c RANKS 12)

add_executable(test_cos_transform  test_fft3d.h test_cos.cpp)
target_link_libraries(test_cos_transform  Heffte)
foreach(_heffte_num_procs 1 2 4)
    heffte_add_mpi_test(NAME test_cos_np${_heffte_num_procs}   COMMAND  test_cos_transform RANKS ${_heffte_num_procs})
endforeach()
unset(_heffte_num_procs)

# test the 32 and 64 bit integers
add_executable(test_longlong  test_fft3d.h test_longlong.cpp)
target_link_libraries(test_longlong  Heffte)
heffte_add_mpi_test(NAME heffte_longlong_np4   COMMAND test_longlong RANKS 4)

if (CMAKE_C_COMPILER)
    add_executable(test_c_binding  test_c.c)
    target_link_libraries(test_c_binding  Heffte)
    heffte_add_mpi_test(NAME heffte_bindc_np2   COMMAND test_c_binding RANKS 2)
endif()

if (Heffte_ENABLE_TRACING)
    add_executable(test_trace  test_trace.cpp)
    target_link_libraries(test_trace  Heffte)
    heffte_add_mpi_test(NAME test_tracing   COMMAND test_trace RANKS 2)
endif()
