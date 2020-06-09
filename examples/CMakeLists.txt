
if (Heffte_ENABLE_FFTW)
    add_executable(heffte_example_fftw heffte_example_fftw.cpp)
    target_link_libraries(heffte_example_fftw Heffte)

    add_executable(heffte_example_options heffte_example_options.cpp)
    target_link_libraries(heffte_example_options Heffte)

    add_executable(heffte_example_vectors heffte_example_vectors.cpp)
    target_link_libraries(heffte_example_vectors Heffte)
endif()
