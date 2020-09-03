
execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE heffte_cxx_version)
string(REGEX MATCH "^HIP" heffte_haship "${heffte_cxx_version}")

if (NOT heffte_haship)
    message(WARNING "Tasmanian_ENABLE_ROCM requires that the CMAKE_CXX_COMPILER is set to the Rocm hipcc compiler.")
endif()

get_filename_component(heffte_hipccroot ${CMAKE_CXX_COMPILER} DIRECTORY)
get_filename_component(heffte_hipccroot ${heffte_hipccroot} DIRECTORY)

set(Heffte_ROCM_ROOT "${heffte_hipccroot}" CACHE PATH "The root folder for the Rocm framework installation")
list(APPEND CMAKE_PREFIX_PATH "${Heffte_ROCM_ROOT}")

find_package(rocfft REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HeffteRocFFT DEFAULT_MSG ROCFFT_LIBRARIES)
