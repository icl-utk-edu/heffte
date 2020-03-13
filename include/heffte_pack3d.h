/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_PACK3D_H
#define HEFFTE_PACK3D_H

#include <string.h>
#include "heffte_common.h"

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

namespace heffte {
// identical to before, but in this namespace
struct pack_plan_3d{
    int nfast;                 // # of elements in fast index
    int nmid;                  // # of elements in mid index
    int nslow;                 // # of elements in slow index
    int line_stride;           // stride between successive mid indices
    int plane_stride;          // stride between successive slow indices
};

inline std::ostream & operator << (std::ostream &os, pack_plan_3d const &plan){
    os << "nfast = " << plan.nfast << "\n";
    os << "nmid  = " << plan.nmid << "\n";
    os << "nslow = " << plan.nslow << "\n";
    os << "line_stride = "  << plan.line_stride << "\n";
    os << "plane_stride = " << plan.plane_stride << "\n";
    os << "\n";
    return os;
}

/*!
 * \brief The packer needs to know whether the data will be on the CPU or GPU devices.
 *
 * Specializations of this template will define the type alias \b mode
 * that will be set to either the tag::cpu or tag::gpu.
 */
template<typename backend>
struct packer_backend{};

// typename struct packer_backend<cuda>{ using mode = tag::gpu; } // specialization can differentiate between gpu and cpu backends

/*!
 * \brief Defines the direct packer without implementation, use the specializations to get the CPU or GPU implementation.
 */
template<typename mode> struct direct_packer{};

/*!
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::cpu>{
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        scalar_type* buffer_iterator = buffer;
        for(int slow = 0; slow < plan.nslow; slow++){
            for(int mid = 0; mid < plan.nmid; mid++){
                buffer_iterator = std::copy_n(&data[slow * plan.plane_stride + mid * plan.line_stride], plan.nfast, buffer_iterator);
            }
        }
    }
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        for(int slow = 0; slow < plan.nslow; slow++){
            for(int mid = 0; mid < plan.nmid; mid++){
                std::copy_n(&buffer[(slow * plan.nmid + mid) * plan.nfast],
                            plan.nfast, &data[slow * plan.plane_stride + mid * plan.line_stride]);
            }
        }
    }
};

}

#endif
