#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "heffte_utils.h"

// Magma  ALL2ALL options
enum algo_magma_a2av_type_t{
ALL2ALLV  = 0,                   // MPI_Alltoallv (Default)
MAGMA_A2AV  = 1,                 // MPI_Isend + MPI_Irecv + selfcopy
ALL2ALLV_SC = 2,                 // MPI_Alltoallv + self cudaMemcpy
IA2AV = 3,                       // MPI_Ialltoallv
SCATTER_GATHER = 4,              // MPI_Scatterv + MPI_Gatherv
SCATTER_GATHER_SC = 5,           // MPI_Scatterv + MPI_Gatherv
IPC_VERSION = 6,                 // Optimization via IPC communication
};

template <class T>
void magma_Alltoallv(T *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, T *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm, algo_magma_a2av_type_t algo);

// Memory
static const unsigned mem_aligned = (1 << HEFFTE_MEM_CPU_ALIGN) | (1 << HEFFTE_MEM_REG_ALIGN) | (1 << HEFFTE_MEM_MANAGED_ALIGN);

namespace HEFFTE_NS {
  class Memory {
    public:
      enum heffte_memory_type_t memory_type;
      Memory() {}
      void *smalloc(int64_t, heffte_memory_type_t);
      void *srealloc(void *, int64_t, heffte_memory_type_t);
      void sfree(void *, heffte_memory_type_t );
  };
}


// Error

namespace HEFFTE_NS {
class Error {
 public:
  Error(MPI_Comm);
  void all(const char *);
  void one(const char *);
  void warning(const char *);

 private:
  MPI_Comm world;
};
}

// Scale
template <class T>
void scale_ffts_gpu(int n, T *data, T fnorm);

#endif   //  #ifndef HEFFTE_COMMON_H
