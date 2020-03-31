#
# Benchmarks
#
# include dirs

include_directories(${CMAKE_SOURCE_DIR}/test)

set(speed3d_cpu_src speed3d_cpu.cpp)
set(speed3d_gpu_src speed3d_gpu.cpp)

add_executable(speed3d_cpu ${speed3d_cpu_src})
add_executable(speed3d_gpu ${speed3d_gpu_src})

target_link_libraries(speed3d_cpu Heffte)
target_link_libraries(speed3d_gpu Heffte)