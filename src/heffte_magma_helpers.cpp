/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_config.h"

#ifdef Heffte_ENABLE_MAGMA

#include "heffte_magma_helpers.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include "magma_v2.h"

namespace heffte {

namespace gpu {

magma_handle<tag::gpu>::magma_handle(){
    magma_init();
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("cudaGetDevice() failed with message: ") + cudaGetErrorString(status));
    magma_queue_create_from_cuda(device, nullptr, nullptr, nullptr, reinterpret_cast<magma_queue_t*>(&handle));
}
magma_handle<tag::gpu>::~magma_handle(){
    magma_queue_destroy(reinterpret_cast<magma_queue_t>(handle));
    magma_finalize();
}
void magma_handle<tag::gpu>::scal(int num_entries, double scale_factor, float *data) const{
    magma_sscal(num_entries, static_cast<float>(scale_factor), data, 1, reinterpret_cast<magma_queue_t>(handle));
}
void magma_handle<tag::gpu>::scal(int num_entries, double scale_factor, double *data) const{
    magma_dscal(num_entries, scale_factor, data, 1, reinterpret_cast<magma_queue_t>(handle));
}

}

} // namespace heffte

#endif
