include("${CMAKE_CURRENT_LIST_DIR}/HeffteTargets.cmake")

if (@Heffte_ENABLE_FFTW@ AND NOT TARGET Heffte::FFTW)
    add_library(Heffte::FFTW INTERFACE IMPORTED GLOBAL)
    target_link_libraries(Heffte::FFTW INTERFACE @FFTW_LIBRARIES@)
    set_target_properties(Heffte::FFTW PROPERTIES INTERFACE_INCLUDE_DIRECTORIES @FFTW_INCLUDES@)
endif()

if (@Heffte_ENABLE_MKL@ AND NOT TARGET Heffte::MKL)
    add_library(Heffte::MKL INTERFACE IMPORTED GLOBAL)
    target_link_libraries(Heffte::MKL INTERFACE @Heffte_MKL_LIBRARIES@)
    set_target_properties(Heffte::MKL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES @Heffte_MKL_INCLUDES@)
endif()

if (@Heffte_ENABLE_ROCM@ AND NOT TARGET roc::rocfft)
    if (NOT "@Heffte_ROCM_ROOT@" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "@Heffte_ROCM_ROOT@")
    endif()
    find_package(rocfft REQUIRED)
endif()

if (NOT TARGET MPI::MPI_CXX)
    if (NOT MPI_CXX_COMPILER)
        set(MPI_CXX_COMPILER @MPI_CXX_COMPILER@)
    endif()
    find_package(MPI REQUIRED)
endif()

if ("@BUILD_SHARED_LIBS@")
    set(Heffte_SHARED_FOUND "ON")
else()
    set(Heffte_STATIC_FOUND "ON")
endif()
set(Heffte_FFTW_FOUND    "@Heffte_ENABLE_FFTW@")
set(Heffte_MKL_FOUND     "@Heffte_ENABLE_MKL@")
set(Heffte_CUDA_FOUND    "@Heffte_ENABLE_CUDA@")
set(Heffte_ROCM_FOUND    "@Heffte_ENABLE_ROCM@")
