/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "heffte_utils.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// HEFFTE  ALL2ALL options
enum algo_heffte_a2av_type_t{
ALL2ALLV  = 0,                   // MPI_Alltoallv (Default)
HEFFTE_A2AV  = 1,                 // MPI_Isend + MPI_Irecv + selfcopy
ALL2ALLV_SC = 2,                 // MPI_Alltoallv + self cudaMemcpy
IA2AV = 3,                       // MPI_Ialltoallv
SCATTER_GATHER = 4,              // MPI_Scatterv + MPI_Gatherv
SCATTER_GATHER_SC = 5,           // MPI_Scatterv + MPI_Gatherv
IPC_VERSION = 6,                 // Optimization via IPC communication
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// HEFFTE memory options
enum heffte_memory_type_t{
HEFFTE_MEM_CPU = 0,
HEFFTE_MEM_CPU_ALIGN = 1,
HEFFTE_MEM_REG = 2,
HEFFTE_MEM_REG_ALIGN = 3,
HEFFTE_MEM_MANAGED = 4,
HEFFTE_MEM_MANAGED_ALIGN = 5,
HEFFTE_MEM_GPU = 6,
HEFFTE_MEM_PIN = 7,
};

template <class T>
void heffte_Alltoallv(T *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, T *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm, algo_heffte_a2av_type_t algo);

// Memory
static const unsigned mem_aligned = (1 << HEFFTE_MEM_CPU_ALIGN) | (1 << HEFFTE_MEM_REG_ALIGN) | (1 << HEFFTE_MEM_MANAGED_ALIGN);

namespace HEFFTE {
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

namespace HEFFTE {
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


namespace heffte{

namespace tag{
/*
 * Empty structs do not generate run-time code,
 * but can be used in type checks and overload resolutions at compile time.
 * Such empty classes are called "type-tags".
 */

/*!
 * \brief Indicates the use of cpu backend and that all input/output data and arrays will be bound to the cpu.
 *
 * Examples of cpu backends are FFTW and MKL.
 */
struct cpu{};
/*!
 * \brief Indicates the use of gpu backend and that all input/output data and arrays will be bound to the gpu device.
 *
 * Example of gpu backend is cuFFT.
 */
struct gpu{};

// The backends should sit in the main subspace or in a space called backend (or something).
// Each backend will have a separate file indicating only the necessary methods to interact with the main code.
struct fftw{};
}

}

#endif   //  #ifndef HEFFTE_COMMON_H
