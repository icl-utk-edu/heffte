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


namespace heffte {
namespace cuda {

/*
 * Launch with one thread per entry.
 *
 * If to_complex is true, convert one real number from source to two real numbers in destination.
 * If to_complex is false, convert two real numbers from source to one real number in destination.
 */
template<typename scalar_type, int num_threads, bool to_complex>
__global__ void real_complex_convert(int num_entries, scalar_type const source[], scalar_type destination[]){
    int i = blockIdx.x * num_threads + threadIdx.x;
    while(i < num_entries){
        if (to_complex){
            destination[2*i] = source[i];
            destination[2*i + 1] = 0.0;
        }else{
            destination[i] = source[2*i];
        }
        i += num_threads * gridDim.x;
    }
}

/*
 * Launch this with one block per line.
 */
template<typename scalar_type, int num_threads, int tuple_size, bool pack>
__global__ void direct_packer(int nfast, int nmid, int nslow, int line_stride, int plane_stide,
                                          scalar_type const source[], scalar_type destination[]){
    int block_index = blockIdx.x;
    while(block_index < nmid * nslow){

        int mid = block_index % nmid;
        int slow = block_index / nmid;

        scalar_type const *block_source = (pack) ?
                            &source[tuple_size * (mid * line_stride + slow * plane_stide)] :
                            &source[block_index * nfast * tuple_size];
        scalar_type *block_destination = (pack) ?
                            &destination[block_index * nfast * tuple_size] :
                            &destination[tuple_size * (mid * line_stride + slow * plane_stide)];

        int i = threadIdx.x;
        while(i < nfast * tuple_size){
            block_destination[i] = block_source[i];
            i += num_threads;
        }

        block_index += gridDim.x;
    }
}

/*
 * Create a 1-D CUDA thread grid using the total_threads and number of threads per block.
 * Basically, computes the number of blocks but no more than 65536.
 */
struct thread_grid_1d{
    // Compute the threads and blocks.
    thread_grid_1d(int total_threads, int num_per_block) :
        threads(num_per_block),
        blocks(std::min(total_threads / threads + ((total_threads % threads == 0) ? 0 : 1), 65536))
    {}
    // number of threads
    int const threads;
    // number of blocks
    int const blocks;
};

// max number of cuda threads (Volta supports more, but I don't think it matters)
constexpr int max_threads  = 1024;
// allows expressive calls to_complex or not to_complex
constexpr bool to_complex  = true;
// allows expressive calls to_pack or not to_pack
constexpr bool to_pack     = true;

template<typename precision_type>
void convert(int num_entries, precision_type const source[], std::complex<precision_type> destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, to_complex><<<grid.blocks, grid.threads>>>(num_entries, source, reinterpret_cast<precision_type*>(destination));
}
template<typename precision_type>
void convert(int num_entries, std::complex<precision_type> const source[], precision_type destination[]){
    thread_grid_1d grid(num_entries, max_threads);
    real_complex_convert<precision_type, max_threads, not to_complex><<<grid.blocks, grid.threads>>>(num_entries, reinterpret_cast<precision_type const*>(source), destination);
}

template void convert<float>(int num_entries, float const source[], std::complex<float> destination[]);
template void convert<double>(int num_entries, double const source[], std::complex<double> destination[]);
template void convert<float>(int num_entries, std::complex<float> const source[], float destination[]);
template void convert<double>(int num_entries, std::complex<double> const source[], double destination[]);

/*
 * For float and double, defines type = <float/double> and tuple_size = 1
 * For complex float/double, defines type <float/double> and typle_size = 2
 */
template<typename scalar_type> struct precision{
    using type = scalar_type;
    static const int tuple_size = 1;
};
template<typename precision_type> struct precision<std::complex<precision_type>>{
    using type = precision_type;
    static const int tuple_size = 2;
};

template<typename scalar_type>
void direct_pack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]){
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, to_pack>
            <<<std::min(nmid * nslow, 65536), max_threads>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template<typename scalar_type>
void direct_unpack(int nfast, int nmid, int nslow, int line_stride, int plane_stide, scalar_type const source[], scalar_type destination[]){
    using prec = typename precision<scalar_type>::type;
    direct_packer<prec, max_threads, precision<scalar_type>::tuple_size, not to_pack>
            <<<std::min(nmid * nslow, 65536), max_threads>>>(nfast, nmid, nslow, line_stride, plane_stide,
            reinterpret_cast<prec const*>(source), reinterpret_cast<prec*>(destination));
}

template void direct_pack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_pack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_pack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_pack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

template void direct_unpack<float>(int, int, int, int, int, float const source[], float destination[]);
template void direct_unpack<double>(int, int, int, int, int, double const source[], double destination[]);
template void direct_unpack<std::complex<float>>(int, int, int, int, int, std::complex<float> const source[], std::complex<float> destination[]);
template void direct_unpack<std::complex<double>>(int, int, int, int, int, std::complex<double> const source[], std::complex<double> destination[]);

}
}

#endif
