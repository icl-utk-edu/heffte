########################################################################
# FindHeffteOneApi.cmake module
#######################################################################

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE heffte_cxx_version)
string(FIND "DPC++" heffte_hasdpcpp "${heffte_cxx_version}")

if (heffte_hasdpcpp LESS 0)
    message(WARNING "Heffte_ENABLE_ONEAPI requires that the CMAKE_CXX_COMPILER is set to the Intel dpcpp compiler.")
endif()

get_filename_component(heffte_oneapi_root ${CMAKE_CXX_COMPILER} DIRECTORY)  # convert <path>/bin/dpcpp to <path>/bin
get_filename_component(heffte_oneapi_root ${heffte_oneapi_root} DIRECTORY)  # convert <path>/bin to <path>

set(Heffte_ONEMKL_ROOT "$ENV{MKLROOT}" CACHE PATH "The root folder for the Intel OneMKL framework installation")

heffte_find_libraries(REQUIRED mkl_sycl
                      OPTIONAL mkl_intel_lp64 mkl_intel_thread mkl_core
                      PREFIX ${Heffte_ONEMKL_ROOT}
                      LIST onemkl)

find_package_handle_standard_args(HeffteOneApi DEFAULT_MSG heffte_onemkl)

# create imported target
add_library(Heffte::OneMKL INTERFACE IMPORTED GLOBAL)
target_link_libraries(Heffte::OneMKL INTERFACE ${heffte_onemkl})
