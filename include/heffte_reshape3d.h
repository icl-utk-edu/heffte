/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

/** @class */
// Reshape3d class

#ifndef HEFFTE_RESHAPE3D_H
#define HEFFTE_RESHAPE3D_H

#include <mpi.h>
#include "heffte_utils.h"
#include "heffte_common.h"

namespace HEFFTE {

  /*!
   * The class Reshape3d is in charge of data reshape, starting from the input data to the first
   * direction, and going to every direction to finalize by reshaping the computed FFT into the output
   * shape. Objects can be created as follows: new Reshape3d(MPI_Comm user_comm)
   * @param user_comm  MPI communicator for the P procs which own the data
   */

template <class U>
class Reshape3d {
 public:
  int collective;         // 0 = point-to-point MPI, 1 = collective all2all
  int packflag;           // 0 = array, 1 = pointer, 2 = memcpy
  int64_t memusage;       // memory usage in bytes

  enum heffte_memory_type_t memory_type;

  Reshape3d(MPI_Comm);
  ~Reshape3d();

  void setup(int in_ilo, int in_ihi, int in_jlo, int in_jhi, int in_klo, int in_khi,
             int out_ilo, int out_ihi, int out_jlo, int out_jhi, int out_klo, int out_khi,
             int nqty, int user_permute, int user_memoryflag, int &user_sendsize, int &user_recvsize);

  template <class T>
  void reshape(T *in, T *out, T *user_sendbuf, T *user_recvbuf);

 private:
  MPI_Comm world;
  int me,nprocs,me_newcomm,nprocs_newcomm;
  int setupflag;

  class Memory *memory;
  class Memory *memory_cpu;
  class Error *error;

  // details of how to perform a 3d reshape

  int permute;                      // permutation setting = 0,1,2
  int memoryflag;                   // 0 = user-provided bufs, 1 = internal

  // point to point communication

  int nrecv;                        // # of recvs from other procs
  int nsend;                        // # of sends to other procs
  int self;                         // whether I send/recv with myself

  int *send_offset;                 // extraction loc for each send
  int *send_size;                   // size of each send message
  int *send_proc;                   // proc to send each message to
  struct pack_plan_3d *packplan;    // pack plan for each send message
  int *recv_offset;                 // insertion loc for each recv
  int *recv_size;                   // size of each recv message
  int *recv_proc;                   // proc to recv each message from
  int *recv_bufloc;                 // offset in scratch buf for each recv
  MPI_Request *request;             // MPI request for each posted recv
  struct pack_plan_3d *unpackplan;  // unpack plan for each recv message

  // collective communication

  int ngroup;                       // # of procs in my collective comm group
  int *pgroup;                      // list of ranks in my comm group
  MPI_Comm newcomm;                 // communicator for my comm group

  int *sendcnts;                    // args for MPI_All2all()
  int *senddispls;
  int *sendmap;
  int *recvcnts;
  int *recvdispls;
  int *recvmap;

  // memory for reshape sends and recvs and All2all
  // either provided by caller or allocated internally

  U *sendbuf;              // send buffer
  U *recvbuf;              // recv buffer

  // which pack & unpack functions to use

  void (*pack)(U *, U *, struct pack_plan_3d *);
  void (*unpack)(U *, U *, struct pack_plan_3d *);

  // collision between 2 regions

  struct extent_3d {
    int ilo,ihi,isize;
    int jlo,jhi,jsize;
    int klo,khi,ksize;
  };

  int collide(struct extent_3d *, struct extent_3d *, struct extent_3d *);
};

}

#endif
