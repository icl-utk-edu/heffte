# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#
# The file creates the imported target
#   HEFFTE::FFTW             ... allows liking to the FFTW package
#
# The following variables will be checked by the function
#   FFTW_USE_STATIC_LIBS    ... if true, only static libraries are found
#   FFTW_ROOT               ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW_LIBRARY            ... fftw library to use
#   FFTW_INCLUDE_DIR        ... fftw include directory
#

macro(heffte_find_fftw_libraries)
# Usage:
#   heffte_find_fftw_libraries(PREFIX <fftw-root>
#                              VAR <list-name>
#                              REQUIRED <list-names, e.g., "fftw3" "fftw3f">
#                              OPTIONAL <list-names, e.g., "fftw3_threads">)
#  will append the result from find_library() to the <list-name>
#  both REQUIRED and OPTIONAL libraries will be searched
#  if PREFIX is true, then it will be searched exclusively
#                     otherwise standard paths will be used in the search
#  if a library listed in REQUIRED is not found, a FATAL_ERROR will be raised
#
  cmake_parse_arguments(heffte_fftw "" "PREFIX;VAR" "REQUIRED;OPTIONAL" ${ARGN})
  foreach(_lib ${heffte_fftw_REQUIRED} ${heffte_fftw_OPTIONAL})
    if (heffte_fftw_PREFIX)
      find_library(
          heffte_fftw_lib
          NAMES "${_lib}"
          PATHS ${heffte_fftw_PREFIX}
          PATH_SUFFIXES "lib" "lib64"
          NO_DEFAULT_PATH
      )
    else()
      # not sure what LIB_INSTALL_DIR means
      find_library(
          heffte_fftw_lib
          NAMES "${_lib}"
          PATHS ${PKG_FFTW_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
      )
    endif()
    if (heffte_fftw_lib)
      list(APPEND ${heffte_fftw_VAR} ${heffte_fftw_lib})
      unset(heffte_fftw_lib CACHE)
    elseif (${heffte_fftw_lib} IN_LIST ${heffte_fftw_REQUIRED})
      message(FATAL_ERROR "Could not find required fftw3 library: ${heffte_fftw_lib}")
    endif()
  endforeach()
  unset(_lib)
endmacro(heffte_find_fftw_libraries)

#If environment variable FFTW_ROOT_DIR is specified, it has same effect as FFTW_ROOT
if( NOT FFTW_ROOT AND ENV{FFTW_ROOT_DIR} )
  set( FFTW_ROOT $ENV{FFTW_ROOT_DIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

#Determine from PKG
if( PKG_CONFIG_FOUND AND NOT FFTW_ROOT )
  pkg_check_modules( PKG_FFTW QUIET "fftw3" )
endif()

#Check whether to search static or dynamic libs
set( CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES} )

if( ${FFTW_USE_STATIC_LIBS} )
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
else()
  set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX} )
endif()

heffte_find_fftw_libraries(
    PREFIX ${FFTW_ROOT}
    VAR FFTW_LIBRARIES
    REQUIRED "fftw3" "fftw3f"
    OPTIONAL "fftw3_threads" "fftw3f_threads"
  )

if( FFTW_ROOT )
  #find includes
  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
  )

else()
  find_path(
    FFTW_INCLUDES
    NAMES "fftw3.h"
    PATHS ${PKG_FFTW_INCLUDE_DIRS} ${INCLUDE_INSTALL_DIR}
  )

endif( FFTW_ROOT )

set( CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
                                  FFTW_INCLUDES FFTW_LIBRARIES)

add_library(Heffte::FFTW INTERFACE IMPORTED GLOBAL)
target_link_libraries(Heffte::FFTW INTERFACE ${FFTW_LIBRARIES})
set_target_properties(Heffte::FFTW PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDES})

unset(CMAKE_FIND_LIBRARY_SUFFIXES_SAV)
