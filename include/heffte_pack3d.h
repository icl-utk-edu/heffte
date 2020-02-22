/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_PACK3D_H
#define HEFFTE_PACK3D_H

#include <string.h>

namespace HEFFTE {

struct pack_plan_3d {
  int nfast;                 // # of elements in fast index
  int nmid;                  // # of elements in mid index
  int nslow;                 // # of elements in slow index
  int nstride_line;          // stride between successive mid indices
  int nstride_plane;         // stride between successive slow indices
  int nqty;                  // # of values/element
};


/* ----------------------------------------------------------------------
   Pack and unpack functions:

   pack routines copy strided values from data into contiguous locs in buf
   unpack routines copy contiguous values from buf into strided locs in data
   different versions of unpack depending on permutation
     and # of values/element
   ARRAY routines work via array indices (default)
   POINTER routines work via pointers
   MEMCPY routines work via pointers and memcpy function
------------------------------------------------------------------------- */

// ----------------------------------------------------------------------
// pack/unpack with array indices
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */

template <class T>
 void pack_3d_array(T *data, T *buf,
                          struct pack_plan_3d *plan);
/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_array(T *buf, T *data,
                            struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_1_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_2_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_n_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_1_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_2_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_n_array(T *buf, T *data,
                                       struct pack_plan_3d *plan);

// ----------------------------------------------------------------------
// pack/unpack with pointers
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */

template <class T>
 void pack_3d_pointer(T *data, T *buf,
                            struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_pointer(T *buf, T *data,
                              struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_1_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_2_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_n_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_1_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_2_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_n_pointer(T *buf, T *data,
                                         struct pack_plan_3d *plan);

// ----------------------------------------------------------------------
// pack/unpack with pointers and memcpy function
// no memcpy version of unpack_permute methods, just use POINTER version
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------
   pack from data -> buf
------------------------------------------------------------------------- */

template <class T>
 void pack_3d_memcpy(T *data, T *buf,
                           struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_memcpy(T *buf, T *data,
                             struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_1_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_2_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, one axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute1_n_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 1 value/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_1_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, 2 values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_2_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ----------------------------------------------------------------------
   unpack from buf -> data, two axis permutation, nqty values/element
------------------------------------------------------------------------- */

template <class T>
 void unpack_3d_permute2_n_memcpy(T *buf, T *data,
                                        struct pack_plan_3d *plan);

/* ---------------------------------------------------------------------- */

}

#endif
