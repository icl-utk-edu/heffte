cmake_minimum_required(VERSION 3.10)

project("Heffte" VERSION 1.0.1 LANGUAGES CXX)

option(Heffte_ENABLE_FFTW     "Enable the FFTW backend"            OFF)
option(Heffte_ENABLE_CUDA     "Enable the CUDA and cuFFT backend"  OFF)
option(Heffte_ENABLE_ROCM     "Enable the HIP and rocFFT backend"  OFF)
option(Heffte_ENABLE_MKL      "Enable the Intel MKL backend"       OFF)
option(Heffte_ENABLE_DOXYGEN  "Build the Doxygen documentation"    OFF)

option(Heffte_ENABLE_MAGMA    "Enable some helper functions from UTK MAGAM for GPU backends"   OFF)

option(Heffte_ENABLE_TRACING  "Enable the tracing capabilities"  OFF)

if (CMAKE_C_COMPILER)
    enable_language(C)
endif()

# check if at least one backend has been enabled
if (NOT Heffte_ENABLE_FFTW AND
    NOT Heffte_ENABLE_CUDA AND
    NOT Heffte_ENABLE_ROCM AND
    NOT Heffte_ENABLE_MKL
    )
    message(FATAL_ERROR "HeFFTe requires at least one enabled backend!")
endif()

# guard against in-source builds (may be tricked by sym-links, but it is as good as it gets)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed, please perform an out-of-source or out-of-place build, see https://cmake.org/runningcmake/ for details.")
endif()

# Set default install path to build
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "default install path" FORCE )
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/heffte_config.cmake.h"
               "${CMAKE_CURRENT_BINARY_DIR}/include/heffte_config.h")

# find common dependencies
find_package(MPI REQUIRED) # always a dependency

if (Heffte_ENABLE_FFTW OR Heffte_ENABLE_MKL OR Heffte_ENABLE_ROCM)
    # allow cmake to use custom modules (e.g. FindFFTW)
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
endif()

# libheffte source files list (does not include the cuda kernels)
set(Heffte_common_sources
    include/heffte.h
    include/heffte_c.h
    include/heffte_utils.h
    include/heffte_trace.h
    include/heffte_geometry.h
    include/heffte_common.h
    include/heffte_backend_vector.h
    include/heffte_plan_logic.h
    include/heffte_pack3d.h
    include/heffte_reshape3d.h
    include/heffte_fft3d.h
    include/heffte_fft3d_r2c.h
    src/heffte_c.cpp
    src/heffte_plan_logic.cpp
    src/heffte_reshape3d.cpp
    src/heffte_fft3d.cpp
    src/heffte_fft3d_r2c.cpp
    )

if (Heffte_ENABLE_ROCM)
    list(APPEND Heffte_common_sources
         include/heffte_backend_rocm.h
         src/heffte_backend_rocm.hip.cpp
         )
endif()

# build CPU libheffte, the CUDA test comes first to use cuda_add_library() vs add_library()
if (Heffte_ENABLE_CUDA)
    find_package(CUDA REQUIRED)

    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    cuda_add_library(Heffte ${Heffte_common_sources}
                            include/heffte_backend_cuda.h
                            src/heffte_backend_cuda.cu)

    cuda_add_cufft_to_target(Heffte)
    target_include_directories(Heffte PUBLIC $<INSTALL_INTERFACE:${CUDA_INCLUDE_DIRS}>)
else()
    add_library(Heffte ${Heffte_common_sources})
endif()

# handle other dependencies
target_link_libraries(Heffte MPI::MPI_CXX)

if (Heffte_ENABLE_FFTW)
    find_package(HeffteFFTW REQUIRED)
    target_link_libraries(Heffte Heffte::FFTW)
endif()

if (Heffte_ENABLE_MKL)
    find_package(HeffteMKL REQUIRED)
    target_link_libraries(Heffte Heffte::MKL)
endif()

if (Heffte_ENABLE_ROCM)
    find_package(HeffteRocFFT REQUIRED)
    target_link_libraries(Heffte roc::rocfft)
endif()

if (Heffte_ENABLE_MAGMA)
    find_package(HeffteMAGMA REQUIRED)
    list(FILTER CUDA_CUBLAS_LIBRARIES EXCLUDE REGEX "-NOTFOUND$") # work-around CMake 3.10 + CUDA 10
    target_link_libraries(Heffte Heffte::MAGMA ${CUDA_CUBLAS_LIBRARIES})
endif()

# other target properties
target_compile_features(Heffte PUBLIC cxx_std_11)
set_target_properties(Heffte PROPERTIES OUTPUT_NAME "heffte"
                                        CXX_EXTENSIONS OFF
                                        SOVERSION ${Heffte_VERSION_MAJOR}
                                        VERSION   ${PROJECT_VERSION})

# include folders
target_include_directories(Heffte PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)
target_include_directories(Heffte PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>)
target_include_directories(Heffte PUBLIC $<INSTALL_INTERFACE:include>)

###########################
# Documentation
###########################
if (Heffte_ENABLE_DOXYGEN)
    # must come after add_library(Heffte ...)
    add_subdirectory(doxygen)
endif()


###########################
# install
###########################
install(TARGETS Heffte EXPORT Heffte_Targets DESTINATION lib)
install(EXPORT Heffte_Targets FILE HeffteTargets.cmake NAMESPACE Heffte:: DESTINATION lib/cmake/Heffte)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/HeffteConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/HeffteConfigVersion.cmake" DESTINATION lib/cmake/Heffte)
install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h"
                                                              PATTERN "**~" EXCLUDE
                                                              PATTERN "*_gpu*" EXCLUDE
                                                              PATTERN "*.cu*" EXCLUDE
                                                              )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/heffte_config.h DESTINATION include/)

# package-config
include(CMakePackageConfigHelpers)
write_basic_package_version_file("HeffteConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeffteConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/HeffteConfig.cmake" @ONLY)


######################
# EXAMPLES and TESTS #
######################
add_subdirectory(benchmarks)
add_subdirectory(examples)
enable_testing()
add_subdirectory(test)


###########################
# Post Install Test
###########################
# The REGEX helps accept both list and regular set of flags.
string(REGEX REPLACE ";" " " Heffte_mpi_preflags  "${MPIEXEC_PREFLAGS}")
string(REGEX REPLACE ";" " " Heffte_mpi_postflags "${MPIEXEC_POSTFLAGS}")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/post_install_test.cmake.sh" "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh" @ONLY)
add_custom_target(test_install COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh")


##############################
# Examples for post install
##############################
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExampleCMakeLists.cmake" "${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt" DESTINATION share/heffte/examples)
install(DIRECTORY examples/ DESTINATION share/heffte/examples FILES_MATCHING PATTERN "*.cpp")

# print summary of the CMake options, skip if using add_subdirectory(heffte)
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_summary.cmake)
endif()


##############################
# Handles the rpath
##############################
include(cmake/set_rpath.cmake)
