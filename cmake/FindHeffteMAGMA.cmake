# - Find the MAGMA library
#
# Usage:
#   find_package(HeffteMAGMA [REQUIRED] [QUIET] )
#   Heffte::MAGMA target is created (if successful)
#   HeffteMAGMA_LIBRARIES and HeffteMAGMA_INCLUDES will be defined
#   The variables can also be used to bypass the search

set(MAGMA_ROOT "$ENV{MAGMA_ROOT}" CACHE PATH "The root folder for the MAGMA installation, e.g., containing lib and include folders")

# respect the provided libraries
if (NOT HeffteMAGMA_LIBRARIES)
    find_library(HeffteMAGMA_LIBRARIES
                 NAMES "magma"
                 PATHS ${MAGMA_ROOT}
                 PATH_SUFFIXES lib
                 )
endif()

# respect the provided include paths
if (NOT HeffteMAGMA_INCLUDES)
    find_path(HeffteMAGMA_INCLUDES
              NAMES "magma.h"
              PATHS ${MAGMA_ROOT}
              PATH_SUFFIXES "include"
              )
endif()

# handle components and standard CMake arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HeffteMAGMA DEFAULT_MSG
                                  HeffteMAGMA_LIBRARIES HeffteMAGMA_INCLUDES)

# create imported target
add_library(Heffte::MAGMA INTERFACE IMPORTED GLOBAL)
target_link_libraries(Heffte::MAGMA INTERFACE ${HeffteMAGMA_LIBRARIES})
set_target_properties(Heffte::MAGMA PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${HeffteMAGMA_INCLUDES})
