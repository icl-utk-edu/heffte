include("${CMAKE_CURRENT_LIST_DIR}/HEFFTE_Targets.cmake")

if (@Heffte_ENABLE_FFTW@)
    add_library(HEFFTE::FFTW INTERFACE IMPORTED GLOBAL)
    target_link_libraries(HEFFTE::FFTW INTERFACE @FFTW_LIBRARIES@)
    set_target_properties(HEFFTE::FFTW PROPERTIES INTERFACE_INCLUDE_DIRECTORIES @FFTW_INCLUDES@)
endif()

if (NOT TARGET MPI::MPI_CXX)
    if (NOT MPI_CXX_COMPILER)
        set(MPI_CXX_COMPILER @MPI_CXX_COMPILER@)
    endif()
    find_package(MPI REQUIRED)
endif()
