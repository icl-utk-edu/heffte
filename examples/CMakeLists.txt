
if (Heffte_ENABLE_FFTW)
    add_executable(heffte_example_fftw heffte_example_fftw.cpp)
    target_link_libraries(heffte_example_fftw Heffte)
endif()