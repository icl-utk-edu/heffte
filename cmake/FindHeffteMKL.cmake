# - Find the MKL library
#   Using the CMake native find-blas module, but we also need the headers
#

set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "The root folder for the MKL installation")


macro(heffte_find_mkl_libraries)
# Usage:
#   heffte_find_mkl_libraries(PREFIX <fftw-root>
#                             VAR <list-name>
#                             REQUIRED <list-names, e.g., "mkl_cdft_core">
#                             OPTIONAL <list-names, e.g., "mkl_intel_thread">)
#  will append the result from find_library() to the <list-name>
#  both REQUIRED and OPTIONAL libraries will be searched
#  if PREFIX is true, then it will be searched exclusively
#                     otherwise standard paths will be used in the search
#  if a library listed in REQUIRED is not found, a FATAL_ERROR will be raised
#
    cmake_parse_arguments(heffte_fftw "" "PREFIX;VAR" "REQUIRED;OPTIONAL" ${ARGN})
    foreach(heffte_lib ${heffte_fftw_REQUIRED} ${heffte_fftw_OPTIONAL})
        if (heffte_fftw_PREFIX)
            find_library(
                heffte_fftw_lib
                NAMES ${heffte_lib}
                PATHS ${heffte_fftw_PREFIX}
                PATH_SUFFIXES lib
                              lib64
                              lib/intel64
                              ${CMAKE_LIBRARY_ARCHITECTURE}/lib
                              ${CMAKE_LIBRARY_ARCHITECTURE}/lib64
                              lib/${CMAKE_LIBRARY_ARCHITECTURE}
                              lib64/${CMAKE_LIBRARY_ARCHITECTURE}
                NO_DEFAULT_PATH
                        )
        else()
            find_library(
                heffte_fftw_lib
                NAMES ${heffte_lib}
                        )
        endif()
        if (heffte_fftw_lib)
            list(APPEND ${heffte_fftw_VAR} ${heffte_fftw_lib})
        elseif (${heffte_lib} IN_LIST ${heffte_fftw_REQUIRED})
            message(FATAL_ERROR "Could not find required mkl component: ${heffte_lib}")
        endif()
        unset(heffte_fftw_lib CACHE)
    endforeach()
    unset(heffte_lib)
endmacro(heffte_find_mkl_libraries)


heffte_find_mkl_libraries(
        PREFIX ${MKL_ROOT}
        VAR Heffte_MKL_LIBRARIES
        REQUIRED "mkl_intel_ilp64"
        OPTIONAL "mkl_cdft_core" "mkl_intel_thread" "mkl_core" "iomp5"
                               )

foreach(_heffte_sys_path $ENV{LD_LIBRARY_PATH})
    if (NOT Heffte_MKL_IOMP5)
heffte_find_mkl_libraries(
        PREFIX ${_heffte_sys_path}
        VAR Heffte_MKL_IOMP5
        REQUIRED "iomp5"
        OPTIONAL
                               )
    endif()
endforeach()
# message(STATUS "Heffte_MKL_IOMP5 = ${Heffte_MKL_IOMP5}")
list(APPEND Heffte_MKL_LIBRARIES ${Heffte_MKL_IOMP5})
list(APPEND Heffte_MKL_LIBRARIES -liomp5)

foreach(_heffte_blas_lib ${Heffte_MKL_LIBRARIES})
    get_filename_component(_heffte_libpath ${_heffte_blas_lib} DIRECTORY)
    list(APPEND Heffte_mkl_paths ${_heffte_libpath})
endforeach()
unset(_heffte_libpath)
unset(_heffte_blas_lib)

find_path(
    Heffte_MKL_INCLUDES
    NAMES "mkl_dfti.h"
    PATHS ${MKL_ROOT}
    PATH_SUFFIXES "include"
            )

# handle components and standard CMake arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HeffteMKL DEFAULT_MSG
                                  Heffte_MKL_LIBRARIES Heffte_MKL_INCLUDES)

# create imported target
add_library(Heffte::MKL INTERFACE IMPORTED GLOBAL)
target_link_libraries(Heffte::MKL INTERFACE ${Heffte_MKL_LIBRARIES})
set_target_properties(Heffte::MKL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${Heffte_MKL_INCLUDES})
