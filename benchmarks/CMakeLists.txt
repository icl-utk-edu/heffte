#
# Benchmarks
#
# source files list
# include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (speed3d_cpu_src speed3d_cpu.cpp)
add_executable(speed3d_cpu ${speed3d_cpu_src})
target_link_libraries(speed3d_cpu     Heffte)

