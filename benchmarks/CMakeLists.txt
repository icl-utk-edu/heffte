#
# Benchmarks
#

macro(heffte_add_benchmark heffte_benchmark)
  add_executable(${heffte_benchmark} "${heffte_benchmark}.cpp")
  target_link_libraries(${heffte_benchmark} Heffte)
  # using some of the helper methods from the test-modules
  target_include_directories(${heffte_benchmark} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../test/>)
endmacro()

if (Heffte_ENABLE_FFTW)
  heffte_add_benchmark(speed3d_cpu)
endif()
if (Heffte_ENABLE_CUDA)
  heffte_add_benchmark(speed3d_gpu)
endif()