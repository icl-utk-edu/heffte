cmake_minimum_required(VERSION 3.10)

project("Heffte" VERSION 2.1.0 LANGUAGES CXX)

option(Heffte_ENABLE_FFTW     "Enable the FFTW backend"            OFF)
option(Heffte_ENABLE_CUDA     "Enable the CUDA and cuFFT backend"  OFF)
option(Heffte_ENABLE_ROCM     "Enable the HIP and rocFFT backend"  OFF)
option(Heffte_ENABLE_ONEAPI   "Enable the oneAPI/DPC++ and oneMKL backend"  OFF)
option(Heffte_ENABLE_MKL      "Enable the Intel MKL backend"       OFF)
option(Heffte_ENABLE_DOXYGEN  "Build the Doxygen documentation"    OFF)

option(Heffte_ENABLE_AVX      "Enable the use of AVX registers in the stock backend, adds flags: -mfma -mavx"       OFF)
option(Heffte_ENABLE_AVX512   "Enable the use of AVX512 registers in the stock backend, adds AVX flags plus: -mavx512f -mavx512dq"    OFF)

option(Heffte_ENABLE_MAGMA    "Enable some helper functions from UTK MAGMA for GPU backends"   OFF)

option(Heffte_ENABLE_PYTHON   "Configure the Python scripts"     OFF)
option(Heffte_ENABLE_FORTRAN  "Build the Fortran modules for the selected backends." OFF)
option(Heffte_ENABLE_SWIG     "Rebuild the SWIG bindings." OFF)

option(Heffte_ENABLE_TRACING  "Enable the tracing capabilities"  OFF)

option(BUILD_SHARED_LIBS      "Builds shared libraries using CMake conventions"  ON)

# add the file with common CMake helper macros
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/heffte_macros.cmake")

set(Heffte_langs "CXX")
if (CMAKE_C_COMPILER)
    enable_language(C)
    set(Heffte_langs "${Heffte_langs} C")
endif()
if (Heffte_ENABLE_FORTRAN)
    enable_language(Fortran)
    set(Heffte_langs "${Heffte_langs} Fortran")
endif()

# oneAPI requires regular MKL as a CPU backend
if (Heffte_ENABLE_ONEAPI)
    set(Heffte_ENABLE_MKL ON CACHE BOOL "oneAPI requires Intel MKL backend to be enabled" FORCE)
endif()

if (Heffte_ENABLE_AVX512)
    set(Heffte_ENABLE_AVX ON CACHE BOOL "AVX with 256-bits requires AVX512" FORCE)
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

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# libheffte source files list (does not include the cuda kernels)
set(Heffte_common_sources
    include/heffte.h
    include/heffte_c.h
    include/heffte_utils.h
    include/heffte_trace.h
    include/heffte_geometry.h
    include/heffte_common.h
    include/heffte_backend_vector.h
    include/heffte_magma_helpers.h
    include/heffte_plan_logic.h
    include/heffte_pack3d.h
    include/heffte_reshape3d.h
    include/heffte_fft3d.h
    include/heffte_fft3d_r2c.h
    include/heffte_r2r_executor.h
    include/stock_fft/heffte_stock_algos.h
    include/stock_fft/heffte_stock_allocator.h
    include/stock_fft/heffte_stock_complex.h
    include/stock_fft/heffte_stock_tree.h
    include/stock_fft/heffte_stock_vec_types.h
    src/heffte_c.cpp
    src/heffte_plan_logic.cpp
    src/heffte_magma_helpers.cpp
    src/heffte_reshape3d.cpp
    src/heffte_fft3d.cpp
    src/heffte_fft3d_r2c.cpp
    )

if (Heffte_ENABLE_ROCM)
    list(APPEND Heffte_common_sources
         include/heffte_backend_rocm.h
         src/heffte_backend_rocm.hip.cpp
         )

    set(Heffte_ROCM_FLAGS "" CACHE STRING "added flags specific to the rocFFT backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Heffte_ROCM_FLAGS}")
endif()

if (Heffte_ENABLE_ONEAPI)
    list(APPEND Heffte_common_sources
         include/heffte_backend_oneapi.h
         src/heffte_backend_oneapi.cpp
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

if (Heffte_ENABLE_ONEAPI)
    find_package(HeffteOneApi REQUIRED)
    target_link_libraries(Heffte Heffte::OneMKL)
endif()

if (Heffte_ENABLE_MAGMA)
    if (NOT Heffte_ENABLE_CUDA AND NOT Heffte_ENABLE_ROCM)
        message(FATAL_ERROR "MAGMA helpers work only with a GPU backend, e.g., CUDA or ROCM")
    endif()
    find_package(HeffteMAGMA REQUIRED)
    list(FILTER CUDA_CUBLAS_LIBRARIES EXCLUDE REGEX "-NOTFOUND$") # work-around CMake 3.10 + CUDA 10
    target_link_libraries(Heffte Heffte::MAGMA ${CUDA_CUBLAS_LIBRARIES})
endif()

# other target properties
if (Heffte_ENABLE_AVX)
    target_compile_options(Heffte PUBLIC -mfma -mavx)
endif()
if (Heffte_ENABLE_AVX512)
    target_compile_options(Heffte PUBLIC -mavx512f -mavx512dq)
endif()

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
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeffteConfig.cmake"
                              "${CMAKE_CURRENT_BINARY_DIR}/HeffteConfig.cmake"
                              INSTALL_DESTINATION "lib/Heffte/")


######################
# EXAMPLES and TESTS #
######################
add_subdirectory(benchmarks)
add_subdirectory(examples)
enable_testing()
add_subdirectory(test)
if (Heffte_ENABLE_FORTRAN)
    add_subdirectory(fortran)
endif()
if (Heffte_ENABLE_PYTHON)
    add_subdirectory(python)
endif()


###########################
# Post Install Test
###########################
# The REGEX helps accept both list and regular set of flags.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/MakeTest.cmake" "${CMAKE_CURRENT_BINARY_DIR}/configured/CMakeLists.txt" @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/test/post_install_test.cmake.sh" "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh" @ONLY)
add_custom_target(test_install COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/post_install_test.sh")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/configured/CMakeLists.txt"
        DESTINATION "share/heffte/testing/"
        PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)


##############################
# Examples for post install
##############################
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ExampleCMakeLists.cmake" "${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/examples/CMakeLists.txt" DESTINATION share/heffte/examples)
install(DIRECTORY examples/ DESTINATION share/heffte/examples FILES_MATCHING PATTERN "*.cpp" PATTERN "*.c" PATTERN "*.f90")

# print summary of the CMake options, skip if using add_subdirectory(heffte)
if (${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_summary.cmake)
endif()


##############################
# Handles the rpath
##############################
include(cmake/set_rpath.cmake)
