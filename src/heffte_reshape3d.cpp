/**
 * @class
 * CPU functions of HEFFT
 */
 /*
     -- HEFFTE (version 0.2) --
        Univ. of Tennessee, Knoxville
        @date
 */

// Reshape3d class

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "heffte_reshape3d.h"
#include "heffte_pack3d.h"
#include "heffte_trace.h"


namespace HEFFTE {

/*! \fn
 * @param user_comm  MPI communicator for the P procs which own the data
 */
template <class U>
Reshape3d<U>::Reshape3d(MPI_Comm user_comm)
{
  world = user_comm;
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // default settings
  // user can change them before setup()

  collective = 1;
  packflag = 0;

  // Memory and Error classes

  memory = new Memory();

  error = new Error(world);

  // initialize memory allocations

  send_offset = send_size = send_proc = NULL;
  packplan = NULL;

  recv_offset = recv_size = recv_proc = recv_bufloc = NULL;
  request = NULL;
  unpackplan = NULL;

  memusage = 0;
  sendbuf = recvbuf = NULL;

  setupflag = 0;
}
template
Reshape3d<double>::Reshape3d(MPI_Comm user_comm);
template
Reshape3d<float>::Reshape3d(MPI_Comm user_comm);

/* ----------------------------------------------------------------------
   delete a 3d reshape plan
------------------------------------------------------------------------- */
template <class U>
Reshape3d<U>::~Reshape3d()
{
  delete memory;
  delete error;

  // free new MPI communicator for collective comm

  if (collective) {
    if (newcomm != MPI_COMM_NULL) MPI_Comm_free(&newcomm);
    memory->sfree(pgroup,HEFFTE_MEM_CPU_ALIGN);
  }

  // free internal arrays for point-to-point comm
  // also allocated for collective comm

  memory->sfree(send_offset,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(send_size,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(send_proc,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(packplan,HEFFTE_MEM_CPU_ALIGN);

  memory->sfree(recv_offset,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(recv_size,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(recv_proc,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(recv_bufloc,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(request,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(unpackplan,HEFFTE_MEM_CPU_ALIGN);

  // free internal arrays for collective commm

  if (collective) {
    memory->sfree(sendcnts,HEFFTE_MEM_CPU_ALIGN);
    memory->sfree(recvcnts,HEFFTE_MEM_CPU_ALIGN);
    memory->sfree(senddispls,HEFFTE_MEM_CPU_ALIGN);
    memory->sfree(recvdispls,HEFFTE_MEM_CPU_ALIGN);
    memory->sfree(recvmap,HEFFTE_MEM_CPU_ALIGN);
    memory->sfree(sendmap,HEFFTE_MEM_CPU_ALIGN);
  }

  // free buffers if internal

  if (memoryflag) {
    memory->sfree(sendbuf, memory_type);
    memory->sfree(recvbuf, memory_type);
  }
}
template
Reshape3d<double>::~Reshape3d();
template
Reshape3d<float>::~Reshape3d();

/* ----------------------------------------------------------------------
   create plan for performing a 3d reshape

   inputs:
   in_ilo,in_ihi        input bounds of data I own in fast index
   in_jlo,in_jhi        input bounds of data I own in mid index
   in_klo,in_khi        input bounds of data I own in slow index
   out_ilo,out_ihi      output bounds of data I own in fast index
   out_jlo,out_jhi      output bounds of data I own in mid index
   out_klo,out_khi      output bounds of data I own in slow index
   nqty                 # of datums per element
   permute              permutation in storage order of indices on output
                          0 = no permutation
                          1 = permute once = mid->fast, slow->mid, fast->slow
                          2 = permute twice = slow->fast, fast->mid, mid->slow
   memoryflag           user provides buffer memory or system does
                          0 = caller will provide memory
                          1 = system provides memory internally

   outputs:
   sendsize = size of send buffer, caller may choose to provide it
   recvsize = size of recv buffer, caller may choose to provide it
------------------------------------------------------------------------- */

/**
 * Create and setup a plan for performing a 3D reshape of data
 * @param i_lo Integer array of size 3, lower-input bounds of data I own on each of 3 directions
 * @param i_hi Integer array of size 3, upper-input bounds of data I own on each of 3 directions
 * @param o_lo Integer array of size 3, lower-input bounds of data I own on each of 3 directions
 * @param o_hi Integer array of size 3, upper-input bounds of data I own on each of 3 directions
 * @param nqty Number of datums per element
 * @param user_permute Permutation in storage order of indices on output
 * @param user_memoryflag user provides buffer memory (flag=0) or system does (flag=1)
 * @return user_sendsize = Size of send buffer, caller may choose to provide it
 * @return user_recvsize = Size of recv buffer, caller may choose to provide it
 */


 template <class U>
void Reshape3d<U>::setup(int in_ilo, int in_ihi, int in_jlo, int in_jhi,
                    int in_klo, int in_khi,
                    int out_ilo, int out_ihi, int out_jlo, int out_jhi,
                    int out_klo, int out_khi,
                    int nqty, int user_permute, int user_memoryflag,
                    int &user_sendsize, int &user_recvsize)
{
  int i,iproc,ibuf,sendsize,recvsize;
  struct extent_3d in,out,overlap;
  struct extent_3d *inarray,*outarray;

  setupflag = 1;

  permute = user_permute;
  memoryflag = user_memoryflag;

  // store parameters in local data structs

  in.ilo = in_ilo;
  in.ihi = in_ihi;
  in.isize = in.ihi - in.ilo + 1;

  in.jlo = in_jlo;
  in.jhi = in_jhi;
  in.jsize = in.jhi - in.jlo + 1;

  in.klo = in_klo;
  in.khi = in_khi;
  in.ksize = in.khi - in.klo + 1;

  out.ilo = out_ilo;
  out.ihi = out_ihi;
  out.isize = out.ihi - out.ilo + 1;

  out.jlo = out_jlo;
  out.jhi = out_jhi;
  out.jsize = out.jhi - out.jlo + 1;

  out.klo = out_klo;
  out.khi = out_khi;
  out.ksize = out.khi - out.klo + 1;

  // combine output extents across all procs

  inarray = (struct extent_3d *)
    memory->smalloc(nprocs*sizeof(struct extent_3d),HEFFTE_MEM_CPU_ALIGN);
  if (!inarray) error->one("Could not allocate inarray");

  outarray = (struct extent_3d *)
    memory->smalloc(nprocs*sizeof(struct extent_3d),HEFFTE_MEM_CPU_ALIGN);
  if (!outarray) error->one("Could not allocate outarray");

  MPI_Allgather(&out,sizeof(struct extent_3d),MPI_BYTE,
                outarray,sizeof(struct extent_3d),MPI_BYTE,world);

  // count send collides, including self

  nsend = 0;
  iproc = me;
  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    nsend += collide(&in,&outarray[iproc],&overlap);
  }

  // malloc space for send info

  if (nsend) {
    if (packflag == 0) pack = pack_3d_array;
    else if (packflag == 1) pack = pack_3d_pointer;
    else if (packflag == 2) pack = pack_3d_memcpy;
    send_offset = (int *) memory->smalloc(nsend*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    send_size = (int *) memory->smalloc(nsend*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    send_proc = (int *) memory->smalloc(nsend*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    packplan = (struct pack_plan_3d *)
      memory->smalloc(nsend*sizeof(struct pack_plan_3d),HEFFTE_MEM_CPU_ALIGN);
    if (!send_offset || !send_size || !send_proc || !packplan)
      error->one("Could not allocate reshape send info");
  }

  // store send info, with self as last entry

  nsend = 0;
  iproc = me;
  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    if (collide(&in,&outarray[iproc],&overlap)) {
      send_proc[nsend] = iproc;
      send_offset[nsend] = nqty *
        ((overlap.klo-in.klo)*in.jsize*in.isize +
         ((overlap.jlo-in.jlo)*in.isize + overlap.ilo-in.ilo));
      packplan[nsend].nfast = nqty*overlap.isize;
      packplan[nsend].nmid = overlap.jsize;
      packplan[nsend].nslow = overlap.ksize;
      packplan[nsend].nstride_line = nqty*in.isize;
      packplan[nsend].nstride_plane = nqty*in.jsize*in.isize;
      packplan[nsend].nqty = nqty;
      send_size[nsend] = nqty*overlap.isize*overlap.jsize*overlap.ksize;
      nsend++;
    }
  }

  // nsend = # of sends not including self
  // for collective mode include self in nsend list

  if (nsend && send_proc[nsend-1] == me && !collective) nsend--;

  // combine input extents across all procs
  MPI_Allgather(&in,sizeof(struct extent_3d),MPI_BYTE,
                inarray,sizeof(struct extent_3d),MPI_BYTE,world);

  // count recv collides, including self

  nrecv = 0;
  iproc = me;
  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    nrecv += collide(&out,&inarray[iproc],&overlap);
  }

  // malloc space for recv info

  if (nrecv) {
    if (permute == 0) {
      if (packflag == 0) unpack = unpack_3d_array;
      else if (packflag == 1) unpack = unpack_3d_pointer;
      else if (packflag == 2) unpack = unpack_3d_memcpy;
    } else if (permute == 1) {
      if (nqty == 1) {
        if (packflag == 0) unpack = unpack_3d_permute1_1_array;
        else if (packflag == 1) unpack = unpack_3d_permute1_1_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute1_1_memcpy;
      } else if (nqty == 2) {
        if (packflag == 0) unpack = unpack_3d_permute1_2_array;
        else if (packflag == 1) unpack = unpack_3d_permute1_2_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute1_2_memcpy;
      } else {
        if (packflag == 0) unpack = unpack_3d_permute1_n_array;
        else if (packflag == 1) unpack = unpack_3d_permute1_n_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute1_n_memcpy;
      }
    } else if (permute == 2) {
      if (nqty == 1) {
        if (packflag == 0) unpack = unpack_3d_permute2_1_array;
        else if (packflag == 1) unpack = unpack_3d_permute2_1_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute2_1_memcpy;
      } else if (nqty == 2) {
        if (packflag == 0) unpack = unpack_3d_permute2_2_array;
        else if (packflag == 1) unpack = unpack_3d_permute2_2_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute2_2_memcpy;
      } else {
        if (packflag == 0) unpack = unpack_3d_permute2_n_array;
        else if (packflag == 1) unpack = unpack_3d_permute2_n_pointer;
        else if (packflag == 2) unpack = unpack_3d_permute2_n_memcpy;
      }
    }

    recv_offset = (int *) memory->smalloc(nrecv*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    recv_size = (int *) memory->smalloc(nrecv*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    recv_proc = (int *) memory->smalloc(nrecv*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    recv_bufloc = (int *) memory->smalloc(nrecv*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    request = (MPI_Request *) memory->smalloc(nrecv*sizeof(MPI_Request),HEFFTE_MEM_CPU_ALIGN);
    unpackplan = (struct pack_plan_3d *)
      memory->smalloc(nrecv*sizeof(struct pack_plan_3d),HEFFTE_MEM_CPU_ALIGN);
    if (!recv_offset || !recv_size || !recv_proc || !recv_bufloc ||
        !request || !unpackplan)
      error->one("Could not allocate reshape recv info");
  }

  // store recv info, with self as last entry

  ibuf = 0;
  nrecv = 0;
  iproc = me;

  for (i = 0; i < nprocs; i++) {
    iproc++;
    if (iproc == nprocs) iproc = 0;
    if (collide(&out,&inarray[iproc],&overlap)) {
      recv_proc[nrecv] = iproc;
      recv_bufloc[nrecv] = ibuf;

      if (permute == 0) {
        recv_offset[nrecv] = nqty *
          ((overlap.klo-out.klo)*out.jsize*out.isize +
           (overlap.jlo-out.jlo)*out.isize + (overlap.ilo-out.ilo));
        unpackplan[nrecv].nfast = nqty*overlap.isize;
        unpackplan[nrecv].nmid = overlap.jsize;
        unpackplan[nrecv].nslow = overlap.ksize;
        unpackplan[nrecv].nstride_line = nqty*out.isize;
        unpackplan[nrecv].nstride_plane = nqty*out.jsize*out.isize;
        unpackplan[nrecv].nqty = nqty;
      }
      else if (permute == 1) {
        recv_offset[nrecv] = nqty *
          ((overlap.ilo-out.ilo)*out.ksize*out.jsize +
           (overlap.klo-out.klo)*out.jsize + (overlap.jlo-out.jlo));
        unpackplan[nrecv].nfast = overlap.isize;
        unpackplan[nrecv].nmid = overlap.jsize;
        unpackplan[nrecv].nslow = overlap.ksize;
        unpackplan[nrecv].nstride_line = nqty*out.jsize;
        unpackplan[nrecv].nstride_plane = nqty*out.ksize*out.jsize;
        unpackplan[nrecv].nqty = nqty;
      }
      else if (permute == 2) {
        recv_offset[nrecv] = nqty *
          ((overlap.jlo-out.jlo)*out.isize*out.ksize +
           (overlap.ilo-out.ilo)*out.ksize + (overlap.klo-out.klo));
        unpackplan[nrecv].nfast = overlap.isize;
        unpackplan[nrecv].nmid = overlap.jsize;
        unpackplan[nrecv].nslow = overlap.ksize;
        unpackplan[nrecv].nstride_line = nqty*out.ksize;
        unpackplan[nrecv].nstride_plane = nqty*out.isize*out.ksize;
        unpackplan[nrecv].nqty = nqty;
      }

      recv_size[nrecv] = nqty*overlap.isize*overlap.jsize*overlap.ksize;
      ibuf += recv_size[nrecv];
      nrecv++;
    }
  }

  // nrecv = # of recvs not including self
  // for collectives include self in nrecv list

  int nrecv_original = nrecv;
  if (nrecv && recv_proc[nrecv-1] == me && !collective) nrecv--;

  // self = 1 if send/recv data to self

  if (nrecv == nrecv_original) self = 0;
  else self = 1;

  // for point-to-point comm
  // find biggest send message (not including self) and malloc space for it
  // if requested, allocate internal scratch space for recvs,
  // only need it if I will receive any data (including self)

  if (!collective) {
    sendsize = 0;
    for (i = 0; i < nsend; i++) sendsize = std::max(sendsize,send_size[i]);
    recvsize = nqty * out.isize*out.jsize*out.ksize;

    if (memoryflag && sendsize) {
      sendbuf = (U *) memory->smalloc(sendsize*sizeof(U), memory_type);
      if (!sendbuf) error->one("Could not allocate sendbuf array");
    }
    if (memoryflag && recvsize) {
      recvbuf = (U *) memory->smalloc(recvsize*sizeof(U), memory_type);
      if (!recvbuf) error->one("Could not allocate recvbuf array");
    }
  }

  // setup for collective communication
  // pgroup = list of procs I communicate with during reshape
  // ngroup = # of procs in pgroup

  if (collective) {

    // pflag = 1 if proc is in group
    // allocate pgroup as large as all procs

    int *pflag = (int *) memory->smalloc(nprocs*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    for (i = 0; i < nprocs; i++) pflag[i] = 0;

    pgroup = (int *) memory->smalloc(nprocs*sizeof(int),HEFFTE_MEM_CPU_ALIGN);
    ngroup = 0;

    // add procs to pgroup that I send to and recv from, including self

    for (i = 0; i < nsend; i++) {
      if (pflag[send_proc[i]]) continue;
      pflag[send_proc[i]] = 1;
      pgroup[ngroup++] = send_proc[i];
    }

    for (i = 0; i < nrecv; i++) {
      if (pflag[recv_proc[i]]) continue;
      pflag[recv_proc[i]] = 1;
      pgroup[ngroup++] = recv_proc[i];
    }

    // loop over procs in pgroup
    // collide each inarray extent with all Nprocs output extents
    // collide each outarray extent with all Nprocs input extents
    // add any new collision to pgroup
    // keep iterating until nothing is added to pgroup

    int ngroup_extra;

    int active = 1;
    while (active) {
      active = 0;
      ngroup_extra = ngroup;
      for (int i = 0; i < ngroup; i++) {
        iproc = pgroup[i];
        for (int jproc = 0; jproc < nprocs; jproc++) {
          if (pflag[jproc]) continue;
          if (collide(&inarray[iproc],&outarray[jproc],&overlap)) {
            pflag[jproc] = 1;
            pgroup[ngroup_extra++] = jproc;
            active = 1;
          }
          if (pflag[jproc]) continue;
          if (collide(&outarray[iproc],&inarray[jproc],&overlap)) {
            pflag[jproc] = 1;
            pgroup[ngroup_extra++] = jproc;
            active = 1;
          }
        }
      }
      ngroup = ngroup_extra;
    }

    // resize pgroup to final size
    // recreate sorted pgroup from pflag

    pgroup = (int *) memory->srealloc(pgroup,ngroup*sizeof(int),HEFFTE_MEM_CPU_ALIGN);

    ngroup = 0;
    for (i = 0; i < nprocs; i++)
      if (pflag[i]) pgroup[ngroup++] = i;

    memory->sfree(pflag,HEFFTE_MEM_CPU_ALIGN);

    // create all2all communicators for the reshape
    // based on the group each proc belongs to

    MPI_Group orig_group,new_group;
    MPI_Comm_group(world,&orig_group);
    MPI_Group_incl(orig_group,ngroup,pgroup,&new_group);
    MPI_Comm_create(world,new_group,&newcomm);
    MPI_Group_free(&orig_group);
    MPI_Group_free(&new_group);
    MPI_Comm_rank(newcomm,&me_newcomm);
    MPI_Comm_size(newcomm,&nprocs_newcomm);

    // create send and recv buffers for AlltoAllv collective

    sendsize = 0;
    for (int i = 0; i < nsend; i++) sendsize += send_size[i];
    recvsize = 0;
    for (int i = 0; i < nrecv; i++) recvsize += recv_size[i];

    if (memoryflag && sendsize) {
      sendbuf = (U *) memory->smalloc(sendsize*sizeof(U), memory_type);
      if (!sendbuf) error->one("Could not allocate sendbuf array");
    }
    if (memoryflag && recvsize) {
      recvbuf = (U *) memory->smalloc(recvsize*sizeof(U), memory_type);
      if (!recvbuf) error->one("Could not allocate recvbuf array");
    }

    sendcnts = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);
    senddispls = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);
    sendmap = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);
    recvcnts = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);
    recvdispls = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);
    recvmap = (int *) memory->smalloc(sizeof(int)*ngroup,HEFFTE_MEM_CPU_ALIGN);

    if (!sendcnts || !senddispls || !sendmap ||
        !recvcnts || !recvdispls || !recvmap)
      if (ngroup) error->one("Could not allocate all2all args");

    // populate sendcnts and recvdispls vectors
    // order and size of proc group is different than send_proc
    // sendmap[i] = index into send info for Ith proc in pgroup

    int offset = 0;
    for (int isend = 0; isend < ngroup; isend++) {
      sendcnts[isend] = 0;
      senddispls[isend] = 0;
      sendmap[isend] = -1;
      for (int i = 0; i < nsend; i++) {
        if (send_proc[i] != pgroup[isend]) continue;
        sendcnts[isend] = send_size[i];
        senddispls[isend] = offset;
        offset += send_size[i];
        sendmap[isend] = i;
        break;
      }
    }

    // populate recvcnts and recvdispls vectors
    // order and size of proc group is different than recv_proc
    // recvmap[i] = index into recv info for Ith proc in pgroup

    offset = 0;
    for (int irecv = 0; irecv < ngroup; irecv++) {
      recvcnts[irecv] = 0;
      recvdispls[irecv] = 0;
      recvmap[irecv] = -1;
      for (int i = 0; i < nrecv; i++) {
        if (recv_proc[i] != pgroup[irecv]) continue;
        recvcnts[irecv] = recv_size[i];
        recvdispls[irecv] = offset;
        offset += recv_size[i];
        recvmap[irecv] = i;
        break;
      }
    }
  }

  // free allocated extents

  memory->sfree(inarray,HEFFTE_MEM_CPU_ALIGN);
  memory->sfree(outarray,HEFFTE_MEM_CPU_ALIGN);

  // return sizes for send and recv buffers

  user_sendsize = sendsize;
  user_recvsize = recvsize;

  // set memusage
  // note there was also temporary allocation of
  //   inarray,outarray = Nprocs * sizeof(struc extent_3d)

  memusage = 0;

  // allocated for both point-to-point and collective comm
  // 3 send vectors and packplan
  // 4 recv vectors, request, and unpackplan
  // send and recv bufs if caller doesn't allocate them

  memusage += 3*nsend * sizeof(int);
  memusage += nsend * sizeof(struct pack_plan_3d);

  memusage += 4*nrecv * sizeof(int);
  memusage += nrecv * sizeof(MPI_Request *);
  memusage += nrecv * sizeof(struct pack_plan_3d);

  if (memoryflag) {
    memusage += (int64_t) sendsize * sizeof(U);
    memusage += (int64_t) recvsize * sizeof(U);
  }

  // allocated only for collective commm

  if (collective) memusage += 7*ngroup * sizeof(int);
}

template
void Reshape3d<double>::setup(int in_ilo, int in_ihi, int in_jlo, int in_jhi,
                   int in_klo, int in_khi,
                   int out_ilo, int out_ihi, int out_jlo, int out_jhi,
                   int out_klo, int out_khi,
                   int nqty, int user_permute, int user_memoryflag,
                   int &user_sendsize, int &user_recvsize);
template
void Reshape3d<float>::setup(int in_ilo, int in_ihi, int in_jlo, int in_jhi,
                   int in_klo, int in_khi,
                   int out_ilo, int out_ihi, int out_jlo, int out_jhi,
                   int out_klo, int out_khi,
                   int nqty, int user_permute, int user_memoryflag,
                   int &user_sendsize, int &user_recvsize);


/* ----------------------------------------------------------------------
   perform a 3d reshape

   in           starting address of input data on this proc
   out          starting address of where output data for this proc
                  will be placed (can be same as in)
   buf          extra memory required for reshape
                if memoryflag=0 was used in call to setup()
                  user_sendbuf and user_recvbuf are used
                  size was returned to caller by setup()
                if memoryflag=1 was used in call to setup()
                  user_sendbuf and user_recvbuf are not used, can be NULL
------------------------------------------------------------------------- */

/**
 * Perform a 3d Reshape of data
 * @param in Address of input data on this proc
 * @param out Address of output data for this proc (can be same as in)
 * @param user_sendbuf  user allocated memory used if \ref memoryflag was set to 1
 * @param user_recvbuf  user allocated memory used if \ref memoryflag was set to 1
 */
template <class U>
template <class T>
void Reshape3d<U>::reshape(T *in, T *out, T *user_sendbuf, T *user_recvbuf)
{
  int  thread_id = 1;
  char func_name[80], func_message[80];
  int isend,irecv;

  if (!setupflag) error->all("Cannot perform reshape before setup");

  if (!memoryflag) {
    sendbuf = user_sendbuf;
    recvbuf = user_recvbuf;
  }

  // point-to-point reshape communication

  if (!collective) {

    // post all recvs into scratch space
    double t;

    for (irecv = 0; irecv < nrecv; irecv++) {
      snprintf(func_name, sizeof(func_name), "P2P_irecv");
      snprintf(func_message, sizeof(func_message), "P2P_irecv_n%d_s%d",recv_proc[irecv],recv_size[irecv]);
     trace_cpu_start( thread_id, func_name, func_message );

     if(sizeof(T)==4)
      MPI_Irecv(&recvbuf[recv_bufloc[irecv]], recv_size[irecv], MPI_FLOAT, recv_proc[irecv], 0, world, &request[irecv]);
     if(sizeof(T)==8)
      MPI_Irecv(&recvbuf[recv_bufloc[irecv]], recv_size[irecv], MPI_DOUBLE, recv_proc[irecv], 0, world, &request[irecv]);


      trace_cpu_end( thread_id);
    }

    // send all messages to other procs

    for (isend = 0; isend < nsend; isend++) {
      snprintf(func_name, sizeof(func_name), "P2P_pack");
      snprintf(func_message, sizeof(func_message), "P2P_pack_n%d_s%d",send_proc[isend],send_size[isend]);
      trace_cpu_start( thread_id, func_name, func_message );

      t = MPI_Wtime();
      pack(&in[send_offset[isend]],sendbuf,&packplan[isend]);
      #if defined(HEFFTE_TIME_DETAILED)
        timing_array[2] +=  MPI_Wtime() - t;
      #endif

      trace_cpu_end( thread_id);
      snprintf(func_name, sizeof(func_name), "P2P_send");
      snprintf(func_message, sizeof(func_message), "P2P_send_n%d_s%d",send_proc[isend],send_size[isend]);
      trace_cpu_start( thread_id, func_name, func_message );
      if(sizeof(T)==4)
        MPI_Send(sendbuf,send_size[isend],MPI_FLOAT,send_proc[isend],0,world);
      if(sizeof(T)==8)
        MPI_Send(sendbuf,send_size[isend],MPI_DOUBLE,send_proc[isend],0,world);
      trace_cpu_end( thread_id);
    }

    // copy in -> recvbuf -> out for self data

    if (self) {
      isend = nsend;
      snprintf(func_name, sizeof(func_name), "P2P_selfpack");
      snprintf(func_message, sizeof(func_message), "P2Pselfpack");
      trace_cpu_start( thread_id, func_name, func_message );
      t = MPI_Wtime();
      pack(&in[send_offset[isend]],&recvbuf[recv_bufloc[nrecv]],
           &packplan[isend]);
      #if defined(HEFFTE_TIME_DETAILED)
        timing_array[2] +=  MPI_Wtime() - t;
      #endif
      trace_cpu_end( thread_id);

      snprintf(func_name, sizeof(func_name), "P2P_selfunpack");
      snprintf(func_message, sizeof(func_message), "P2Pselfunpack");
      trace_cpu_start( thread_id, func_name, func_message );

      t = MPI_Wtime();
      unpack(&recvbuf[recv_bufloc[nrecv]],&out[recv_offset[nrecv]],
             &unpackplan[nrecv]);
      #if defined(HEFFTE_TIME_DETAILED)
        timing_array[3] += MPI_Wtime() - t;
      #endif
      trace_cpu_end( thread_id);
    }

    // unpack all messages from mybuf -> out

    for (int i = 0; i < nrecv; i++) {
      snprintf(func_name, sizeof(func_name), "P2P_waitany");
      snprintf(func_message, sizeof(func_message), "P2P_waitany%d",i);
      trace_cpu_start( thread_id, func_name, func_message );
      MPI_Waitany(nrecv,request,&irecv,MPI_STATUS_IGNORE);
      trace_cpu_end( thread_id);

      snprintf(func_name, sizeof(func_name), "P2P_unpack");
      snprintf(func_message, sizeof(func_message), "P2P_unpack%d",i);
      trace_cpu_start( thread_id, func_name, func_message );

      t = MPI_Wtime();
      unpack(&recvbuf[recv_bufloc[irecv]],&out[recv_offset[irecv]],
             &unpackplan[irecv]);
      #if defined(HEFFTE_TIME_DETAILED)
        timing_array[3] += MPI_Wtime() - t;
      #endif

      trace_cpu_end( thread_id);
    }

  // All2Allv collective for reshape communication

  } else {

    double t;

    // pack the data into SendBuffer from in
    snprintf(func_name, sizeof(func_name), "A2A_pack");
    snprintf(func_message, sizeof(func_message), "A2A_pack");
    trace_cpu_start( thread_id, func_name, func_message );

    int offset = 0;
    { heffte::add_trace name("packing");
    for (int igroup = 0; igroup < ngroup; igroup++) {
      if (sendmap[igroup] >= 0) {
        isend = sendmap[igroup];

        t = MPI_Wtime();
        pack(&in[send_offset[isend]],&sendbuf[offset],&packplan[isend]);
        #if defined(HEFFTE_TIME_DETAILED)
          timing_array[2] += MPI_Wtime() - t;
        #endif

        offset += send_size[isend];
      }
    }
    }
    trace_cpu_end( thread_id);

// Choose algorithm for all-to-all communication
enum algo_heffte_a2av_type_t HEFFTE_A2AV_algo = ALL2ALLV;

    if (newcomm != MPI_COMM_NULL) {

      #if defined(DTRACING_HEFFTE)
        double avg_snd_siz = 0;
        for (int i = 0; i < ngroup; i++) {
          avg_snd_siz += (sendcnts[i]/1000);
        }
        avg_snd_siz=avg_snd_siz/ngroup;
        snprintf(func_name, sizeof(func_name), "A2A_MPI");
        snprintf(func_message, sizeof(func_message), "A2A_MPI_s%lfk",avg_snd_siz);
        trace_cpu_start( thread_id, func_name, func_message );
      #endif

      t = MPI_Wtime();

      { heffte::add_trace name("all2allv");
      if(sizeof(T)==4)
      heffte_Alltoallv(sendbuf,sendcnts,senddispls,MPI_FLOAT,
                      recvbuf,recvcnts,recvdispls,MPI_FLOAT,
                      newcomm, HEFFTE_A2AV_algo);
      if(sizeof(T)==8)
      heffte_Alltoallv(sendbuf,sendcnts,senddispls,MPI_DOUBLE,
                      recvbuf,recvcnts,recvdispls,MPI_DOUBLE,
                      newcomm, HEFFTE_A2AV_algo);
      }

      #if defined(HEFFTE_TIME_DETAILED)
        timing_array[5] += MPI_Wtime() - t;
      #endif

      #if defined(DTRACING_HEFFTE)
        trace_cpu_end( thread_id);
      #endif
    }


    // unpack the data from recvbuf into out
    snprintf(func_name, sizeof(func_name), "A2A_unpack");
    snprintf(func_message, sizeof(func_message), "A2A_unpack");
    trace_cpu_start( thread_id, func_name, func_message );
    offset = 0;
    { heffte::add_trace name("unpacking");
    for (int igroup = 0; igroup < ngroup; igroup++) {
      if (recvmap[igroup] >= 0) {
        irecv = recvmap[igroup];

        t = MPI_Wtime();
        unpack(&recvbuf[offset],&out[recv_offset[irecv]],&unpackplan[irecv]);
        #if defined(HEFFTE_TIME_DETAILED)
          timing_array[3] += MPI_Wtime() - t;
        #endif

        offset += recv_size[irecv];
      }
    }
    }
    trace_cpu_end( thread_id);
  }
}

template
void Reshape3d<double>::reshape(double *in, double *out,
                    double *user_sendbuf, double *user_recvbuf);
template
void Reshape3d<float>::reshape(float *in, float *out,
                    float *user_sendbuf, float *user_recvbuf);

/* ----------------------------------------------------------------------
   collide 2 sets of indices to determine overlap
   compare bounds of block1 with block2 to see if they overlap
   return 1 if they do and put bounds of overlapping section in overlap
   return 0 if they do not overlap
------------------------------------------------------------------------- */

/**
 * Collides 2 sets of indices to determine overlapping blocks of data from different processors
 * @param block1 Block of indices data on proc 1
 * @param block2 Block of indices data on proc 2
 * @param overlap Block of indices of overlapping data from proc 1 and 2
 * @return 1 if blocks overlap, 0 otherwise
 */
template <class U>
int Reshape3d<U>::collide(struct extent_3d *block1, struct extent_3d *block2,
                     struct extent_3d *overlap)
{
  overlap->ilo = std::max(block1->ilo,block2->ilo);
  overlap->ihi = std::min(block1->ihi,block2->ihi);
  overlap->jlo = std::max(block1->jlo,block2->jlo);
  overlap->jhi = std::min(block1->jhi,block2->jhi);
  overlap->klo = std::max(block1->klo,block2->klo);
  overlap->khi = std::min(block1->khi,block2->khi);

  if (overlap->ilo > overlap->ihi ||
      overlap->jlo > overlap->jhi ||
      overlap->klo > overlap->khi) return 0;

  overlap->isize = overlap->ihi - overlap->ilo + 1;
  overlap->jsize = overlap->jhi - overlap->jlo + 1;
  overlap->ksize = overlap->khi - overlap->klo + 1;

  return 1;
}
template
int Reshape3d<double>::collide(struct extent_3d *block1, struct extent_3d *block2,
                     struct extent_3d *overlap);
template
int Reshape3d<float>::collide(struct extent_3d *block1, struct extent_3d *block2,
                     struct extent_3d *overlap);

}

namespace heffte {

#ifdef Heffte_ENABLE_TRACING

    std::deque<event> event_log;
    std::string log_filename;

#endif


/*!
 * \brief Counts how many boxes from the list have a non-empty intersection with the reference box.
 */
int count_collisions(std::vector<box3d> const &boxes, box3d const reference){
    return std::count_if(boxes.begin(), boxes.end(), [&](box3d const b)->bool{ return not reference.collide(b).empty(); });
}

/*!
 * \brief Returns the ranks that will participate in an all-to-all communication.
 *
 * In a reshape algorithm, consider all ranks and connected them into a graph, where each edge
 * corresponds to a piece of data that must be communicated (send or receive).
 * Then take this rank (defined by the list of send and recv procs) and find the larges connected sub-graph.
 * That corresponds to all the processes that need to participate in an all-to-all communication pattern.
 *
 * \param send_proc is the list of ranks that need data from this rank
 * \param recv_proc is the list of ranks that need to send data to this rank
 * \param input_boxes is the list of all boxes held currently across the comm
 * \param output_boxes is the list of all boxes at the end of the communication
 *
 * \returns a list of ranks that must participate in an all-to-all communication
 */
std::vector<int> a2a_group(std::vector<int> const &send_proc, std::vector<int> const &recv_proc,
                           std::vector<box3d> const &input_boxes, std::vector<box3d> const &output_boxes){
    assert(input_boxes.size() == output_boxes.size());
    std::vector<int> result;
    std::vector<bool> marked(input_boxes.size(), false);

    // start with the processes that are connected to this rank
    for(auto p : send_proc){
        if (marked[p]) continue;
        marked[p] = true;
        result.push_back(p);
    }
    for(auto p : recv_proc){
        if (marked[p]) continue;
        marked[p] = true;
        result.push_back(p);
    }

    // loop over procs in result
    // collide each input_boxes extent with all Nprocs output extents
    // collide each output_boxes extent with all Nprocs input extents
    // add any new collision to result
    // keep iterating until nothing is added to result
    bool adding = true;
    while(adding){
        size_t num_current = result.size();
        for(size_t i=0; i<num_current; i++){
            int iproc = result[i];
            // note the O(n^2) graph search, but should be OK for now
            for(size_t j=0; j<input_boxes.size(); j++){
                if (not marked[j] and not input_boxes[iproc].collide(output_boxes[j]).empty()){
                    result.push_back(j);
                    marked[j] = true;
                }
                if (not marked[j] and not output_boxes[iproc].collide(input_boxes[j]).empty()){
                    result.push_back(j);
                    marked[j] = true;
                }
            }
        }
        adding = (num_current != result.size()); // if nothing got added
    }

    // sort based on the flag
    result.resize(0);
    for(size_t i=0; i<input_boxes.size(); i++)
        if (marked[i]) result.push_back(i);

    return result;
}

void compute_overlap_map(int me, int nprocs, box3d const source, std::vector<box3d> const &boxes,
                         std::vector<int> &proc, std::vector<int> &offset, std::vector<int> &sizes, std::vector<pack_plan_3d> &plans){
    for(int i=0; i<nprocs; i++){
        int iproc = (i + me) % nprocs;
        box3d overlap = source.collide(boxes[iproc]);
        if (not overlap.empty()){
            proc.push_back(iproc);
            offset.push_back((overlap.low[2] - source.low[2]) * source.size[0] * source.size[1]
                              + (overlap.low[1] - source.low[1]) * source.size[0]
                              + (overlap.low[0] - source.low[0]));

            plans.push_back({overlap.size[0], overlap.size[1], overlap.size[2], // fast, mid, and slow sizes
                             source.size[0], source.size[1] * source.size[0]}); // line and plane strides
            sizes.push_back(overlap.count());
        }
    }
}

template<typename backend_tag, template<typename device> class packer>
reshape3d_alltoallv<backend_tag, packer>::reshape3d_alltoallv(
                        int input_size, int output_size,
                        MPI_Comm master_comm, std::vector<int> const &pgroup,
                        std::vector<int> &&csend_offset, std::vector<int> &&csend_size, std::vector<int> const &send_proc,
                        std::vector<int> &&crecv_offset, std::vector<int> &&crecv_size, std::vector<int> const &recv_proc,
                        std::vector<pack_plan_3d> &&cpackplan, std::vector<pack_plan_3d> &&cunpackplan
                                                                ) :
    reshape3d_base(input_size, output_size),
    comm(mpi::new_comm_form_group(pgroup, master_comm)), me(mpi::comm_rank(comm)), nprocs(mpi::comm_size(comm)),
    send_offset(std::move(csend_offset)), send_size(std::move(csend_size)),
    recv_offset(std::move(crecv_offset)), recv_size(std::move(crecv_size)),
    send_total(std::accumulate(send_size.begin(), send_size.end(), 0)),
    recv_total(std::accumulate(recv_size.begin(), recv_size.end(), 0)),
    packplan(std::move(cpackplan)), unpackplan(std::move(cunpackplan)),
    send(pgroup, send_proc, send_size),
    recv(pgroup, recv_proc, recv_size)
{}

template<typename backend_tag, template<typename device> class packer>
template<typename scalar_type>
void reshape3d_alltoallv<backend_tag, packer>::apply_base(scalar_type const source[], scalar_type destination[], scalar_type workspace[]) const{

    using buffer_container = typename backend::buffer_traits<backend_tag>::template container<scalar_type>;

    scalar_type *send_buffer = workspace;
    scalar_type *recv_buffer = workspace + input_size;

    packer<typename backend::buffer_traits<backend_tag>::location> packit;

    int offset = 0;

    { add_trace name("packing");
    for(auto isend : send.map){
        if (isend >= 0){ // something to send
            packit.pack(packplan[isend], source + send_offset[isend], send_buffer + offset);
            offset += send_size[isend];
        }
    }
    }

    #ifdef Heffte_ENABLE_CUDA
    // the device_synchronize() is needed to flush the kernels of the asynchronous packing
    if (std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::gpu>::value)
        cuda::synchronize_default_stream();
    #endif

    { add_trace name("all2allv");
    MPI_Alltoallv(send_buffer, send.counts.data(), send.displacements.data(), mpi::type_from<scalar_type>(),
                  recv_buffer, recv.counts.data(), recv.displacements.data(), mpi::type_from<scalar_type>(),
                  comm);
    }

    offset = 0;
    { add_trace name("unpacking");
    for(auto irecv : recv.map){
        if (irecv >= 0){ // something received
            packit.unpack(unpackplan[irecv], recv_buffer + offset, destination + recv_offset[irecv]);
            offset += recv_size[irecv];
        }
    }
    }
}

template<typename backend_tag, template<typename device> class packer>
std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>
make_reshape3d_alltoallv(std::vector<box3d> const &input_boxes,
                         std::vector<box3d> const &output_boxes,
                         MPI_Comm const comm){
    // if the input and output are the same, returns an empty reshape
    if (match(input_boxes, output_boxes))
        return std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>();

    int const me = mpi::comm_rank(comm);
    int const nprocs = mpi::comm_size(comm);

    std::vector<pack_plan_3d> packplan, unpackplan; // will be moved into the class
    std::vector<int> send_offset;
    std::vector<int> send_size;
    std::vector<int> send_proc;
    std::vector<int> recv_offset;
    std::vector<int> recv_size;
    std::vector<int> recv_proc;

    box3d outbox = output_boxes[me];
    box3d inbox  = input_boxes[me];

    // number of ranks that need data from me
    int nsend = count_collisions(output_boxes, inbox);

    if (nsend > 0) // if others need something from me, prepare the corresponding sizes and plans
        compute_overlap_map(me, nprocs, input_boxes[me], output_boxes, send_proc, send_offset, send_size, packplan);

    // number of ranks that I need data from
    int nrecv = count_collisions(input_boxes, outbox);

    if (nrecv > 0) // if I need something from others, prepare the corresponding sizes and plans
        compute_overlap_map(me, nprocs, output_boxes[me], input_boxes, recv_proc, recv_offset, recv_size, unpackplan);

    return std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>(new reshape3d_alltoallv<backend_tag, packer>(
        inbox.count(), outbox.count(),
        comm, a2a_group(send_proc, recv_proc, input_boxes, output_boxes),
        std::move(send_offset), std::move(send_size), send_proc,
        std::move(recv_offset), std::move(recv_size), recv_proc,
        std::move(packplan), std::move(unpackplan)
                                                       ));
}

#define heffte_instantiate_reshape3d_alltoallv(some_backend) \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<float>(float const[], float[], float[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<double>(double const[], double[], double[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<std::complex<float>>(std::complex<float> const[], std::complex<float>[], std::complex<float>[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<std::complex<double>>(std::complex<double> const[], std::complex<double> [], std::complex<double> []) const; \
 \
template std::unique_ptr<reshape3d_alltoallv<some_backend, direct_packer>> \
make_reshape3d_alltoallv<some_backend, direct_packer>(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const); \


#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_reshape3d_alltoallv(backend::fftw);
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_reshape3d_alltoallv(backend::mkl);
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_reshape3d_alltoallv(backend::cufft);
#endif

}
