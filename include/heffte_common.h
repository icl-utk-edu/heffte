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
#include "heffte_geometry.h"
#include "heffte_trace.h"

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


namespace heffte {

/*!
 * \brief Contains internal type-tags.
 *
 * Empty structs do not generate run-time code,
 * but can be used in type checks and overload resolutions at compile time.
 * Such empty classes are called "type-tags".
 */
namespace tag {

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

}

/*!
 * \brief Contains methods for data manipulation either on the CPU or GPU.
 */
template<typename location_tag> struct data_manipulator{};

/*!
 * \brief Data manipulations on the CPU end.
 */
template<> struct data_manipulator<tag::cpu>{
    /*!
     * \brief Wrapper around std::copy_n().
     */
    template<typename scalar_type>
    static void copy_n(scalar_type const source[], size_t num_entries, scalar_type destination[]){
        std::copy_n(source, num_entries, destination);
    }
    /*!
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void scale(int num_entries, scalar_type *data, double scale_factor){;
        for(int i=0; i<num_entries; i++) data[i] *= scale_factor;
    }
    /*!
     * \brief Complex by real scaling.
     *
     * Depending on the compiler and type of operation, C++ complex numbers can have bad
     * performance compared to float and double operations.
     * Since the scaling factor is always real, scaling can be performed
     * with real arithmetic which is easier to vectorize.
     */
    template<typename precision_type>
    static void scale(int num_entries, std::complex<precision_type> *data, double scale_factor){
        scale<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

/*!
 * \brief Contains type tags and templates metadata for the various backends.
 */
namespace backend {

    /*!
     * \brief Allows to define whether a specific backend interface has been enabled.
     *
     * Defaults to std::false_type, but specializations for each enabled backend
     * will overwrite this to the std::true_type, i.e., define const static bool value
     * which is set to true.
     */
    template<typename tag>
    struct is_enabled : std::false_type{};


    /*!
     * \brief Defines the container for the temporary buffers.
     *
     * Specialization for each backend will define whether the raw-arrays are associated
     * with the CPU or GPU devices and the type of the container that will hold temporary
     * buffers.
     */
    template<typename backend_tag>
    struct buffer_traits{
        //! \brief Tags the raw-array location tag::cpu or tag::gpu, used by the packers.
        using location = tag::cpu;
        //! \brief Defines the container template to use for the temporary buffers in heffte::fft3d.
        template<typename T> using container = std::vector<T>;
    };

    /*!
     * \brief Returns the human readable name of the backend.
     */
    template<typename backend_tag>
    inline std::string name(){ return "unknown"; }
}

/*!
 * \brief Indicates the direction of the FFT.
 */
enum class direction {
    //! \brief Forward DFT transform.
    forward,
    //! \brief Inverse DFT transform.
    backward
};

/*!
 * \brief Indicates the structure that will be used by the fft backend.
 */
template<typename> struct one_dim_backend{};

}

#endif   //  #ifndef HEFFTE_COMMON_H
