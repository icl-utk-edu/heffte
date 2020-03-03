/**
 * @file
 * Pack3d defines the plans for packing/unpacking data to be send/received between processors
 */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

// 3d pack/unpack library

#ifndef FFT_PACK3D_H
#define FFT_PACK3D_H

#include <string.h>
#include "heffte_pack3d.h"
#include "heffte_utils.h"

namespace HEFFTE {

  #define NX 16
  #define NY 16

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */

/**
 * Pack data into buffer using arrays, pflag=array
 * @param data Address of data on this proc
 * @param buf Buffer to store packed data
 * @param plan Pack plan for data packing
*/

template <class T>
void pack_3d_array(T *data, T *buf,
                   struct pack_plan_3d *plan)
{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  in = 0;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_plane;
    for (mid = 0; mid < nmid; mid++) {
      out = plane + mid*nstride_line;
      for (fast = 0; fast < nfast; fast++)
        buf[in++] = data[out++];
    }
  }
}
template
void pack_3d_array(double *data, double *buf,
                   struct pack_plan_3d *plan);
template
void pack_3d_array(float *data, float *buf,
                   struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */
/**
 * Unpack data into buffer using arrays, pflag=array
 * @param buf Buffer to store packed data
 * @param data Address of data on this proc
 * @param plan Pack plan for data packing
*/
template <class T>
void unpack_3d_array(T *buf, T *data,
                     struct pack_plan_3d *plan)
{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_plane;
    for (mid = 0; mid < nmid; mid++) {
      in = plane + mid*nstride_line;
      for (fast = 0; fast < nfast; fast++)
        data[in++] = buf[out++];
    }
  }
}
template
void unpack_3d_array(double *buf, double *data,
                     struct pack_plan_3d *plan);
template
void unpack_3d_array(float *buf, float *data,
                     struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_1_array(T *buf, T *data,
                                struct pack_plan_3d *plan)
{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      in = plane + mid;
      for (fast = 0; fast < nfast; fast++, in += nstride_plane)
        data[in] = buf[out++];
    }
  }
}

template
void unpack_3d_permute1_1_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute1_1_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_2_array(T *buf, T *data,
                                struct pack_plan_3d *plan)
{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      in = plane + 2*mid;
      for (fast = 0; fast < nfast; fast++, in += nstride_plane) {
        data[in] = buf[out++];
        data[in+1] = buf[out++];
      }
    }
  }
}

template
void unpack_3d_permute1_2_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute1_2_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_n_array(T *buf, T *data,
                                struct pack_plan_3d *plan)

{
  register int in,out,iqty,instart,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      instart = plane + nqty*mid;
      for (fast = 0; fast < nfast; fast++, instart += nstride_plane) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) data[in++] = buf[out++];
      }
    }
  }
}

template
void unpack_3d_permute1_n_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute1_n_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_1_array(T *buf, T *data,
                                struct pack_plan_3d *plan)

{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      in = slow + mid*nstride_plane;
      for (fast = 0; fast < nfast; fast++, in += nstride_line)
        data[in] = buf[out++];
    }
  }
}

template
void unpack_3d_permute2_1_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute2_1_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_2_array(T *buf, T *data,
                                struct pack_plan_3d *plan)

{
  register int in,out,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      in = 2*slow + mid*nstride_plane;
      for (fast = 0; fast < nfast; fast++, in += nstride_line) {
        data[in] = buf[out++];
        data[in+1] = buf[out++];
      }
    }
  }
}

template
void unpack_3d_permute2_2_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute2_2_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_n_array(T *buf, T *data,
                                struct pack_plan_3d *plan)
{
  register int in,out,iqty,instart,fast,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = 0;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      instart = nqty*slow + mid*nstride_plane;
      for (fast = 0; fast < nfast; fast++, instart += nstride_line) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) data[in++] = buf[out++];
      }
    }
  }
}

template
void unpack_3d_permute2_n_array(double *buf, double *data,
                                struct pack_plan_3d *plan);
template
void unpack_3d_permute2_n_array(float *buf, float *data,
                                struct pack_plan_3d *plan);

// ----------------------------------------------------------------------
// pack/unpack with pointers
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */
/**
 * Pack data into buffer using arrays, pflag=pointer
 * @param data Address of data on this proc
 * @param buf Buffer to store packed data
 * @param plan Pack plan for data packing
*/
template <class T>
void pack_3d_pointer(T *data, T *buf,
                     struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  in = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_plane;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+mid*nstride_line]);
      end = begin + nfast;
      for (out = begin; out < end; out++)
        *(in++) = *out;
    }
  }
}

template
void pack_3d_pointer(double *data, double *buf,
                     struct pack_plan_3d *plan);
template
void pack_3d_pointer(float *data, float *buf,
                     struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */
/**
 * Unpack data into buffer using arrays, pflag=pointer
 * @param buf Buffer to store packed data
 * @param data Address of data on this proc
 * @param plan Pack plan for data packing
*/
template <class T>
void unpack_3d_pointer(T *buf, T *data,
                       struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_plane;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+mid*nstride_line]);
      end = begin + nfast;
      for (in = begin; in < end; in++)
        *in = *(out++);
    }
  }
}
template
void unpack_3d_pointer(double *buf, double *data,
                       struct pack_plan_3d *plan);
template
void unpack_3d_pointer(float *buf, float *data,
                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_1_pointer(T *buf, T *data,
                                  struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+mid]);
      end = begin + nfast*nstride_plane;
      for (in = begin; in < end; in += nstride_plane)
        *in = *(out++);
    }
  }
}

template
void unpack_3d_permute1_1_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute1_1_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_2_pointer(T *buf, T *data,
                                  struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+2*mid]);
      end = begin + nfast*nstride_plane;
      for (in = begin; in < end; in += nstride_plane) {
        *in = *(out++);
        *(in+1) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute1_2_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute1_2_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_n_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan)
{
  register T *in,*out,*instart,*begin,*end;
  register int iqty,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+nqty*mid]);
      end = begin + nfast*nstride_plane;
      for (instart = begin; instart < end; instart += nstride_plane) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) *(in++) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute1_n_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute1_n_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_1_pointer(T *buf, T *data,
                                  struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (in = begin; in < end; in += nstride_line)
        *in = *(out++);
    }
  }
}
template
void unpack_3d_permute2_1_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute2_1_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_2_pointer(T *buf, T *data,
                                  struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[2*slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (in = begin; in < end; in += nstride_line) {
        *in = *(out++);
        *(in+1) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute2_2_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute2_2_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_n_pointer(T *buf, T *data,
                                  struct pack_plan_3d *plan)
{
  register T *in,*out,*instart,*begin,*end;
  register int iqty,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[nqty*slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (instart = begin; instart < end; instart += nstride_line) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) *(in++) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute2_n_pointer(double *buf, double *data,
                                  struct pack_plan_3d *plan);
template
void unpack_3d_permute2_n_pointer(float *buf, float *data,
                                  struct pack_plan_3d *plan);

// ----------------------------------------------------------------------
// pack/unpack with pointers and memcpy function
// no memcpy version of unpack_permute methods, just use POINTER version
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */
/**
 * Pack data into buffer using arrays, pflag=memcpy
 * @param data Address of data on this proc
 * @param buf Buffer to store packed data
 * @param plan Pack plan for data packing
*/

#if defined(FFT_CUFFTW) || defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

  template <class T>
  __global__ void pack_3d_memcpy_kernel(T *data, T *buf,
  	                              int nfast, int nstride_plane, int nstride_line){
    int fast = threadIdx.x;
    int slow = blockIdx.x;
    int mid  = blockIdx.y,  nmid  = gridDim.y ;

    int plane = slow*nstride_plane;
    int upto  = slow*nmid*nfast;

    data += plane+mid*nstride_line;
    buf  += upto+mid*nfast;

    for(int i=fast; i<nfast; i+= blockDim.x)
       buf[i] = data[i];
  }


  template <class T>
  void pack_3d_memcpy(T * data, T * buf,
                             struct pack_plan_3d *plan)

  {
    int nfast,nmid,nslow,nstride_line,nstride_plane;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    dim3 grid(nslow, nmid);
    dim3 threads(512);
    pack_3d_memcpy_kernel<<<grid, threads>>>(data, buf, nfast, nstride_plane, nstride_line);
    heffte_check_cuda_error();
    cudaDeviceSynchronize();
  }


#else

  template <class T>
  void pack_3d_memcpy(T *data, T *buf,
                      struct pack_plan_3d *plan)
  {
    register T *in,*out;
    register int mid,slow,size;
    register int nfast,nmid,nslow,nstride_line,nstride_plane,plane,upto;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    size = nfast*sizeof(T);
    for (slow = 0; slow < nslow; slow++) {
      plane = slow*nstride_plane;
      upto = slow*nmid*nfast;
      for (mid = 0; mid < nmid; mid++) {
        in = &(buf[upto+mid*nfast]);
        out = &(data[plane+mid*nstride_line]);
        memcpy(in,out,size);
      }
    }
  }

#endif

template
void pack_3d_memcpy(double *data, double *buf,
                    struct pack_plan_3d *plan);
template
void pack_3d_memcpy(float *data, float *buf,
                    struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */
/**
 * Unpack data into buffer using arrays, pflag=memcpy
 * @param buf Buffer to store packed data
 * @param data Address of data on this proc
 * @param plan Pack plan for data packing
*/

#if defined(FFT_CUFFTW) || defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

  template <class T>
  __global__ void unpack_3d_memcpy_kernel(T *data, T *buf,
                                          int nfast, int nstride_plane, int nstride_line){
    int fast = threadIdx.x;
    int slow = blockIdx.x;
    int mid  = blockIdx.y,  nmid  = gridDim.y ;

    int plane = slow*nstride_plane;
    int upto  = slow*nmid*nfast;

    data += plane+mid*nstride_line;
    buf  += upto+mid*nfast;

    for (int i=fast; i<nfast; i+=blockDim.x)
       data[i] = buf[i];
  }

  template <class T>
  void unpack_3d_memcpy(T *buf, T *data,
                               struct pack_plan_3d *plan)

  {
    register int nfast,nmid,nslow,nstride_line,nstride_plane;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    dim3 grid(nslow, nmid);
    dim3 threads(512);
    unpack_3d_memcpy_kernel<<<grid, threads>>>(data, buf, nfast, nstride_plane, nstride_line);
    heffte_check_cuda_error();
    cudaDeviceSynchronize();
  }

#else

  template <class T>
  void unpack_3d_memcpy(T *buf, T *data,
                               struct pack_plan_3d *plan)
  {
    register T *in,*out;
    register int mid,slow,size;
    register int nfast,nmid,nslow,nstride_line,nstride_plane,plane,upto;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    size = nfast*sizeof(T);
    for (slow = 0; slow < nslow; slow++) {
      plane = slow*nstride_plane;
      upto = slow*nmid*nfast;
      for (mid = 0; mid < nmid; mid++) {
        in = &(data[plane+mid*nstride_line]);
        out = &(buf[upto+mid*nfast]);
        memcpy(in,out,size);
      }
    }
  }

#endif

template
void unpack_3d_memcpy(double *buf, double *data,
                      struct pack_plan_3d *plan);
template
void unpack_3d_memcpy(float *buf, float *data,
                      struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_1_memcpy(T *buf, T *data,
                                 struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+mid]);
      end = begin + nfast*nstride_plane;
      for (in = begin; in < end; in += nstride_plane)
        *in = *(out++);
    }
  }
}

template
void unpack_3d_permute1_1_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute1_1_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

#if defined(FFT_CUFFTW) || defined(FFT_CUFFT) || defined(FFT_CUFFT_M) || defined(FFT_CUFFT_R)

  template <class T>
  __global__ void unpack_3d_permute1_2_memcpy_kernel(T *data, T *buf,
                                                     int nmid, int nfast, int nstride_plane, int nstride_line){
    int tx   = threadIdx.x/2, ty = threadIdx.y, tz = threadIdx.x%2;
    int slow = blockIdx.x;
    int mid  = blockIdx.y * NX;
    int fast = blockIdx.z * NY;

    int plane = slow*nstride_line;

    if (mid +tx < nmid && fast + ty < nfast) {
       data += plane+2*mid + fast*nstride_plane;
       buf  += 2*(fast + mid*nfast + slow*nfast*nmid);

       data[2*tx + ty*nstride_plane +tz] = buf[2*ty + 2*tx*nfast +tz];
    }
  }

  template <class T>
  void unpack_3d_permute1_2_memcpy(T *buf, T *data,
                                          struct pack_plan_3d *plan)

  {
    register int nfast,nmid,nslow,nstride_line,nstride_plane;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    dim3 grid(nslow, fft_ceildiv(nmid, NX), fft_ceildiv(nfast, NY));
    dim3 threads(2*NX, NY);
    unpack_3d_permute1_2_memcpy_kernel<<<grid, threads>>>(data, buf, nmid, nfast, nstride_plane, nstride_line);
    heffte_check_cuda_error();
    cudaDeviceSynchronize();
  }

#else

  template <class T>
  void unpack_3d_permute1_2_memcpy(T *buf, T *data,
                                   struct pack_plan_3d *plan)
  {
    register T *in,*out,*begin,*end;
    register int mid,slow;
    register int nfast,nmid,nslow,nstride_line,nstride_plane,plane;

    nfast = plan->nfast;
    nmid = plan->nmid;
    nslow = plan->nslow;
    nstride_line = plan->nstride_line;
    nstride_plane = plan->nstride_plane;

    out = buf;
    for (slow = 0; slow < nslow; slow++) {
      plane = slow*nstride_line;
      for (mid = 0; mid < nmid; mid++) {
        begin = &(data[plane+2*mid]);
        end = begin + nfast*nstride_plane;
        for (in = begin; in < end; in += nstride_plane) {
          *in = *(out++);
          *(in+1) = *(out++);
        }
      }
    }
  }

#endif

template
void unpack_3d_permute1_2_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute1_2_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute1_n_memcpy(T *buf, T *data,
                                 struct pack_plan_3d *plan)
{
  register T *in,*out,*instart,*begin,*end;
  register int iqty,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    plane = slow*nstride_line;
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[plane+nqty*mid]);
      end = begin + nfast*nstride_plane;
      for (instart = begin; instart < end; instart += nstride_plane) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) *(in++) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute1_n_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute1_n_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_1_memcpy(T *buf, T *data,
                                 struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (in = begin; in < end; in += nstride_line)
        *in = *(out++);
    }
  }
}

template
void unpack_3d_permute2_1_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute2_1_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_2_memcpy(T *buf, T *data,
                                 struct pack_plan_3d *plan)
{
  register T *in,*out,*begin,*end;
  register int mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[2*slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (in = begin; in < end; in += nstride_line) {
        *in = *(out++);
        *(in+1) = *(out++);
      }
    }
  }
}
template
void unpack_3d_permute2_2_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute2_2_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
void unpack_3d_permute2_n_memcpy(T *buf, T *data,
                                 struct pack_plan_3d *plan)
{
  register T *in,*out,*instart,*begin,*end;
  register int iqty,mid,slow;
  register int nfast,nmid,nslow,nstride_line,nstride_plane,nqty;

  nfast = plan->nfast;
  nmid = plan->nmid;
  nslow = plan->nslow;
  nstride_line = plan->nstride_line;
  nstride_plane = plan->nstride_plane;
  nqty = plan->nqty;

  out = buf;
  for (slow = 0; slow < nslow; slow++) {
    for (mid = 0; mid < nmid; mid++) {
      begin = &(data[nqty*slow+mid*nstride_plane]);
      end = begin + nfast*nstride_line;
      for (instart = begin; instart < end; instart += nstride_line) {
        in = instart;
        for (iqty = 0; iqty < nqty; iqty++) *(in++) = *(out++);
      }
    }
  }
}

template
void unpack_3d_permute2_n_memcpy(double *buf, double *data,
                                 struct pack_plan_3d *plan);
template
void unpack_3d_permute2_n_memcpy(float *buf, float *data,
                                 struct pack_plan_3d *plan);

/* ---------------------------------------------------------------------- */

}

template<typename scalar_type, int num_threads>
__global__ void heffte_cufft_convert_to_complex(int num_entries, scalar_type const source[], scalar_type destination[]){
    int i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        destination[2*i] = source[i];
        destination[2*i + 1] = 0.0;
        i += num_threads * gridDim.x;
    }
}
template<typename scalar_type, int num_threads>
__global__ void heffte_cufft_convert_to_real(int num_entries, scalar_type const source[], scalar_type destination[]){
    int i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        destination[i] = source[2*i];
        i += num_threads * gridDim.x;
    }
}

namespace heffte {
namespace cuda {

struct thread_grid_1d{
    thread_grid_1d(int total_threads, int num_per_block) :
        threads(num_per_block),
        blocks(clamp(total_threads / threads + ((total_threads % threads == 0) ? 0 : 1)))
    {}
    static int clamp(int candidate_blocks){ return (candidate_blocks >= 65536) ? 65536 : candidate_blocks; }
    int const threads;
    int const blocks;
};

constexpr int max_threads  = 1024;

void convert(int num_entries, float const source[], std::complex<float> destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    heffte_cufft_convert_to_complex<float, max_threads><<<grid.blocks, grid.threads>>>(num_entries, source, reinterpret_cast<float*>(destination));
}
void convert(int num_entries, double const source[], std::complex<double> destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    heffte_cufft_convert_to_complex<double, max_threads><<<grid.blocks, grid.threads>>>(num_entries, source, reinterpret_cast<double*>(destination));
}

void convert(int num_entries, std::complex<float> const source[], float destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    heffte_cufft_convert_to_real<float, max_threads><<<grid.blocks, grid.threads>>>(num_entries, reinterpret_cast<float const*>(source), destination);
}
void convert(int num_entries, std::complex<double> const source[], double destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    heffte_cufft_convert_to_real<double, max_threads><<<grid.blocks, grid.threads>>>(num_entries, reinterpret_cast<double const*>(source), destination);
}

}
}

#endif
