
if (Heffte_ENABLE_FFTW)
    add_executable(heffte_example_fftw heffte_example_fftw.cpp)
    target_link_libraries(heffte_example_fftw Heffte)

    add_executable(heffte_example_options heffte_example_options.cpp)
    target_link_libraries(heffte_example_options Heffte)

    add_executable(heffte_example_vectors heffte_example_vectors.cpp)
    target_link_libraries(heffte_example_vectors Heffte)

    add_executable(heffte_example_r2c heffte_example_r2c.cpp)
    target_link_libraries(heffte_example_r2c Heffte)
endif()

if (Heffte_ENABLE_CUDA)
    add_executable(heffte_example_cuda heffte_example_cuda.cpp)
    target_link_libraries(heffte_example_cuda Heffte)
endif()
