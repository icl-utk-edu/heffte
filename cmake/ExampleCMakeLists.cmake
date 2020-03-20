cmake_minimum_required(VERSION 3.10)

project("HeffteExamples" VERSION @PROJECT_VERSION@ LANGUAGES CXX)

find_package(Heffte @PROJECT_VERSION@ REQUIRED PATHS "@CMAKE_INSTALL_PREFIX@")

if (Heffte_FFTW_FOUND)
    add_executable(heffte_example_fftw heffte_example_fftw.cpp)
    target_link_libraries(heffte_example_fftw Heffte::Heffte)
endif()
