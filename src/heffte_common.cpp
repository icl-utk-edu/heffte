/**
 * @class
 * CPU functions of HEFFT
 */
 /*
     -- HEFFTE (version 0.1) --
        Univ. of Tennessee, Knoxville
        @date
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "heffte_common.h"

#if defined(__INTEL_COMPILER)
#if defined(FFT_USE_TBB_ALLOCATOR)
#include "tbb/scalable_allocator.h"
#else
#include <malloc.h>
#endif
#endif

// #define FFT_PRINTF printf
#define FFT_PRINTF(...)

#if !defined(FFT_MEMALIGN)
#define FFT_MEMALIGN 64
#endif

//Number of bytes we're using for storing the aligned pointer offset
typedef uint16_t offset_t;
#define PTR_OFFSET_SZ sizeof(offset_t)
#ifndef align_up
#define align_up(num, align) \
    (((num) + ((align) - 1)) & ~((align) - 1))
#endif

using namespace HEFFTE_NS;

// Magma ALL2ALL  functions
template <class T>
void magma_Alltoallv(T *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, T *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm, algo_magma_a2av_type_t algo)
{
 int me, nprocs, j=0;
 MPI_Comm_rank(comm,&me);
 MPI_Comm_size(comm,&nprocs);

  switch (algo) {
     case ALL2ALLV:{
          MPI_Alltoallv(sendbuf,sendcounts,sdispls,recvtype, recvbuf,recvcounts,rdispls,recvtype, comm);
          break;
        }

     case MAGMA_A2AV:{
          MPI_Request request[2*nprocs];
          for(int neighbor=0; neighbor<nprocs; neighbor++){
                if(me!=neighbor){
                  MPI_Irecv(recvbuf+rdispls[neighbor], recvcounts[neighbor], recvtype, neighbor, neighbor, comm, request + j);
                  j++;
                }
          }

          for(int neighbor=0; neighbor<nprocs; neighbor++){
                if(me!=neighbor){
                  MPI_Isend(sendbuf+sdispls[neighbor], sendcounts[neighbor], sendtype, neighbor, me, comm, request + j);
                  j++;
                }
                else{
                  #ifdef FFT_CUFFT_A
                  cudaMemcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype),  cudaMemcpyDeviceToDevice);
                  #else
                  memcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype));
                  #endif
                }
          }
          MPI_Waitall(j, request, MPI_STATUSES_IGNORE);
          break;
        }


     case ALL2ALLV_SC:{
           for(int neighbor=0; neighbor<nprocs; neighbor++){
                 if(me==neighbor){
                   #ifdef FFT_CUFFT_A
                   cudaMemcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype),  cudaMemcpyDeviceToDevice);
                   #else
                   memcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype));
                   #endif
                 }
               }
           if(nprocs>1)
                MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, sendtype, comm);
           break;
         }


     case IA2AV:{
          MPI_Request request_t;
          MPI_Status status;
          MPI_Ialltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, sendtype, comm, &request_t);
          MPI_Barrier(comm);
          MPI_Wait(&request_t,&status);
          break;}



    case SCATTER_GATHER:{
          for(int root=0; root<nprocs; root++){
              MPI_Scatterv(sendbuf,  sendcounts, sdispls, sendtype, recvbuf+rdispls[root], recvcounts[root], recvtype, root, comm);
              MPI_Gatherv(sendbuf+sdispls[root],  sendcounts[root], sendtype, recvbuf, recvcounts, rdispls,  recvtype, root, comm);
              }
          break;
        }


    case SCATTER_GATHER_SC:{
          for(int neighbor=0; neighbor<nprocs; neighbor++)
                if(me==neighbor){
                  #ifdef FFT_CUFFT_A
                  cudaMemcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype),  cudaMemcpyDeviceToDevice);
                  #else
                  memcpy(recvbuf + rdispls[me], sendbuf + sdispls[me],  sendcounts[me]*sizeof(sendtype));
                  #endif
            }

         for(int root=0; root<nprocs; root++){
             MPI_Scatterv(sendbuf,  sendcounts, sdispls, sendtype, recvbuf+rdispls[root], recvcounts[root], recvtype, root, comm);
             MPI_Gatherv(sendbuf+sdispls[root],  sendcounts[root], sendtype, recvbuf, recvcounts, rdispls,  recvtype, root, comm);
              }
          break;
        }



    case IPC_VERSION:{
          #ifdef FFT_CUFFT_A
          cudaIpcMemHandle_t *memHandles = new cudaIpcMemHandle_t[nprocs];
          cudaIpcGetMemHandle(memHandles + me, sendbuf);

          // Getting memHandles from all processors
          MPI_Allgather(memHandles + me, sizeof(cudaIpcMemHandle_t), MPI_BYTE, memHandles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm);

          cudaDeviceSynchronize();
          MPI_Barrier(comm);

          for(int neighbor=0; neighbor<nprocs; neighbor++){
            if(me==neighbor){
              // Self copy
                cudaMemcpy(recvbuf + rdispls[neighbor], sendbuf + sdispls[neighbor],  sendcounts[neighbor]*sizeof(sendtype),  cudaMemcpyDeviceToDevice);
            } else{
              // Open the neighbor's memory handle so we can do a cudaMemcpy
                double *sourcePtr;
                cudaIpcOpenMemHandle((void**)&sourcePtr, memHandles[neighbor], cudaIpcMemLazyEnablePeerAccess);
                cudaMemcpy(recvbuf+rdispls[neighbor], sourcePtr + sdispls[neighbor], sendcounts[neighbor]*sizeof(sendtype), cudaMemcpyDefault);
                cudaIpcCloseMemHandle(sourcePtr);
            }
            cudaDeviceSynchronize();
          }

          delete [] memHandles;
          #else
                  printf("Error, routine magma_Alltoallv with IPC communicaton is unavailable for CPU data\n");
                  exit(EXIT_FAILURE);
          #endif
          break;
        }

    default:
          exit(EXIT_SUCCESS);
          break;
}

}

template
void magma_Alltoallv(double *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, double *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm, algo_magma_a2av_type_t algo);
template
void magma_Alltoallv(float *sendbuf, const int *sendcounts,
                     const int *sdispls, MPI_Datatype sendtype, float *recvbuf,
                     const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                     MPI_Comm comm, algo_magma_a2av_type_t algo);

// Memory functions

void * aligned_cuda_malloc(size_t align, size_t size)
{
    void * ptr = NULL;

    //We want it to be a power of two since align_up operates on powers of two
    assert((align & (align - 1)) == 0);

    if(align && size)
    {
        /*
         * We know we have to fit an offset value
         * We also allocate extra bytes to ensure we can meet the alignment
         */
        uint32_t hdr_size = PTR_OFFSET_SZ + (align - 1);
        //void * p = malloc(size + hdr_size);
        void *p=NULL;
        #if defined(FFT_CUFFT_A)
        //cudaMalloc( (void**)&p, size + hdr_size); DO NOT WORK SEE SMALLOC FOR COMMENTS
        #elif defined(FFT_CUFFT_M)
        cudaMallocManaged( (void**)&p, size + hdr_size);
	magma_check_cuda_error();
        #endif

        if(p)
        {
            /*
             * Add the offset size to malloc's pointer (we will always store that)
             * Then align the resulting value to the arget alignment
             */
            ptr = (void *) align_up(((uintptr_t)p + PTR_OFFSET_SZ), align);

            //Calculate the offset and store it behind our aligned pointer
            *((offset_t *)ptr - 1) = (offset_t)((uintptr_t)ptr - (uintptr_t)p);

        } // else NULL, could not malloc
    } //else NULL, invalid arguments

    return ptr;
}

void aligned_cuda_free(void * ptr)
{
    assert(ptr);

    /*
    * Walk backwards from the passed-in pointer to get the pointer offset
    * We convert to an offset_t pointer and rely on pointer math to get the data
    */
    offset_t offset = *((offset_t *)ptr - 1);

    /*
    * Once we have the offset, we can get our original pointer and call free
    */
    void * p = (void *)((uint8_t *)ptr - offset);
    #if defined(FFT_CUFFT_A)
    //cudaFree(p); DO NOT WORK SEE SMALLOC FOR COMMENTS
    #elif defined(FFT_CUFFT_M)
    cudaFree(p);
    #endif
}

/**
 * Allocate memory
 * @param nbytes Number of bytets to allocate
 * @param memory_type Type of memory to allocate, HEFFT provides several options
 */
void *Memory::smalloc(int64_t nbytes, heffte_memory_type_t memory_type)
{
    if (nbytes == 0) return NULL;
    void *ptr=NULL;

    int retval;

    switch (memory_type) {
        case HEFFTE_MEM_GPU:
        printf("we are aboout to allocate gpu \n");

            cudaMalloc((void**)&ptr, nbytes);
            magma_check_cuda_error();
            if (ptr == NULL) printf("null ------------------ \n");

            FFT_PRINTF("FFT_CUFFT_A::: Called allocation called from %s \n", __func__);
            break;

        case HEFFTE_MEM_MANAGED_ALIGN:
            ptr = aligned_cuda_malloc(FFT_MEMALIGN, nbytes);
            magma_check_cuda_error();
            FFT_PRINTF("FFT_CUFFT_M::: Called allocation called from %s \n", __func__);
            break;
        case HEFFTE_MEM_MANAGED:
            cudaMallocManaged((void**)&ptr, nbytes);
            magma_check_cuda_error();
            break;

        case HEFFTE_MEM_REG_ALIGN:
            retval = posix_memalign(&ptr,FFT_MEMALIGN,nbytes);
            if (retval) ptr = NULL;
            if (ptr) { cudaHostRegister(ptr,nbytes,0); magma_check_cuda_error();}
            FFT_PRINTF("FFT_CUFFT_R/USE_CUFFTW::: Called cudaHostRegister allocation called from %s \n", __func__);
            break;
        case HEFFTE_MEM_REG:
            ptr = malloc(nbytes);
            if (ptr) { cudaHostRegister(ptr,nbytes,0); magma_check_cuda_error();}
            FFT_PRINTF("HEFFTE_MEM_REG::: Called allocation called from %s \n", __func__);
            break;

        case HEFFTE_MEM_CPU_ALIGN:
            #if defined(FFT_USE_TBB_ALLOCATOR)
              ptr = scalable_aligned_malloc(nbytes,FFT_MEMALIGN);
            #else
              // FFT_PRINTF("FFT_CPU_MEMALIGN::: Called allocation called from %s \n", __func__);
              retval = posix_memalign(&ptr,FFT_MEMALIGN,nbytes);
              if (retval) ptr = NULL;
            #endif
            break;
        case HEFFTE_MEM_CPU:
            // printf("we are aboout to allocate CPU \n");
            ptr = malloc(nbytes);
            if (ptr == NULL) printf("null ------------------ \n");
            FFT_PRINTF("HEFFTE_MEM_CPU::: Called allocation called from %s \n", __func__);
            break;

        case HEFFTE_MEM_PIN:
            cudaMallocHost((void**)&ptr, nbytes);
            magma_check_cuda_error();
            FFT_PRINTF("HEFFTE_MEM_PIN::: Called allocation called from %s \n", __func__);
            break;

        default:
            exit(EXIT_SUCCESS);
            break;
    }
    return ptr;
}

/**
 * Re-allocate memory
 * @param ptr Pointer to current memory address
 * @param nbytes Number of bytets to reallocate
 * @param memory_type Type of memory to allocate, HEFFT provides several options
 */
void *Memory::srealloc(void *ptr, int64_t nbytes, heffte_memory_type_t memory_type)
{
  if (nbytes == 0) {
    sfree(ptr, memory_type);
    return NULL;
  }

  void *old_ptr = ptr;
  int retval;


  switch (memory_type) {
      case HEFFTE_MEM_GPU:
          cudaMalloc((void**)&ptr, nbytes);
          magma_check_cuda_error();
          if (ptr) cudaMemcpy(ptr, old_ptr, nbytes, cudaMemcpyDeviceToDevice);
          cudaFree(ptr);
          FFT_PRINTF("Realloc FFT_CUFFT_A %s \n",__func__);
          break;

      case HEFFTE_MEM_MANAGED_ALIGN:
          ptr = aligned_cuda_malloc(FFT_MEMALIGN, nbytes);
          magma_check_cuda_error();
          if (ptr) cudaMemcpy(ptr, old_ptr, nbytes, cudaMemcpyDeviceToDevice);
          aligned_cuda_free(old_ptr);
          FFT_PRINTF("Realloc FFT_CUFFT_M %s \n",__func__);
          break;
      case HEFFTE_MEM_MANAGED:
          cudaMallocManaged((void**)&ptr, nbytes);
          magma_check_cuda_error();
          if (ptr) cudaMemcpy(ptr, old_ptr, nbytes, cudaMemcpyDeviceToDevice);
          cudaFree(ptr);
          break;

      case HEFFTE_MEM_REG_ALIGN:
          retval = posix_memalign(&ptr,FFT_MEMALIGN,nbytes);
          if (retval) ptr = NULL;
          if (ptr) cudaHostRegister(ptr,nbytes,0);
          if (ptr) cudaMemcpy(ptr, old_ptr, nbytes, cudaMemcpyDefault);//it could be memcpy as well
          free(old_ptr);
          FFT_PRINTF("Realloc FFT_CUFFT_R/USE_CUFFTW %s \n",__func__);
          break;
      case HEFFTE_MEM_REG:
          cudaHostUnregister(ptr);
          ptr = realloc(ptr, nbytes);
          if (ptr) cudaHostRegister(ptr, nbytes, 0);
          break;

      case HEFFTE_MEM_CPU_ALIGN:
          #if defined(FFT_USE_TBB_ALLOCATOR)
          ptr = scalable_aligned_realloc(ptr, nbytes, FFT_MEMALIGN);
          FFT_PRINTF("%s FFT_USE_TBB \n",__func__);
          #else
          // void *old_ptr = ptr;
          retval = posix_memalign(&ptr,FFT_MEMALIGN,nbytes);
          if (retval) ptr = NULL;
          if (ptr) memcpy(ptr, old_ptr, nbytes);
          FFT_PRINTF("%s FFT_CPU_MEMALIGN \n",__func__);
          #endif
          break;
      case HEFFTE_MEM_CPU:
          ptr = realloc(ptr, nbytes);
          break;

      case HEFFTE_MEM_PIN:
          break;

      default:
          exit(EXIT_SUCCESS);
          break;
  }

    return ptr;
}


/**
 * Deallocate memory
 * @param ptr Pointer to current memory address
 * @param memory_type Type of memory to allocate, HEFFT provides several options
 */
void Memory::sfree(void *ptr, heffte_memory_type_t memory_type)
{
    if (ptr == NULL) return;


    switch (memory_type) {
        case HEFFTE_MEM_GPU:
            cudaFree(ptr);
            magma_check_cuda_error();
            break;

        case HEFFTE_MEM_MANAGED_ALIGN:
            aligned_cuda_free(ptr);
            break;
        case HEFFTE_MEM_MANAGED:
            cudaFree(ptr);
            magma_check_cuda_error();
            break;

        case HEFFTE_MEM_REG_ALIGN:
            cudaHostUnregister(ptr);
            free(ptr);
            break;
        case HEFFTE_MEM_REG:
            cudaHostUnregister(ptr);
            free(ptr);
            break;

        case HEFFTE_MEM_CPU_ALIGN:
            #if defined(FFT_USE_TBB_ALLOCATOR)
              scalable_aligned_free(ptr);
            #else
              free(ptr);
            #endif
            break;
        case HEFFTE_MEM_CPU:
            free(ptr);
            break;

        case HEFFTE_MEM_PIN:
            cudaFreeHost(ptr);
            magma_check_cuda_error();
            break;

        default:
            exit(EXIT_SUCCESS);
            break;
    }

    ptr = NULL;
}

// Error handlers

/* ---------------------------------------------------------------------- */
/*! \fn
 * Define an error handler
 * @param user_comm  MPI communicator for the P procs which own the data
 */
Error::Error(MPI_Comm world_caller)
{
  world = world_caller;
}

void Error::all(const char *str)
{
  MPI_Barrier(world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me == 0) printf("ERROR: %s\n",str);
  MPI_Finalize();
  exit(1);
}

void Error::one(const char *str)
{
  int me;
  MPI_Comm_rank(world,&me);
  printf("ERROR on proc %d: %s\n",me,str);
  MPI_Abort(world,1);
}

void Error::warning(const char *str)
{
  printf("WARNING: %s\n",str);
}
