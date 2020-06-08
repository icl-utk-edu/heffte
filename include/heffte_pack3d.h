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

/*!
 * \ingroup fft3d
 * \addtogroup hefftepacking Packing/Unpacking operations
 *
 * MPI communications assume that the data is located in contiguous arrays;
 * however, the blocks that need to be transmitted in an FFT algorithm
 * correspond to sub-boxes of a three dimensional array, which is never contiguous.
 * Thus, packing and unpacking operations are needed to copy the sub-box into contiguous arrays.
 * Furthermore, some backends (e.g., fftw3) work much faster with contiguous FFT
 * transforms, thus it is beneficial to transpose the data between backend calls.
 * Combing unpack and transpose operations reduces data movement.
 */

namespace heffte {

/*!
 * \ingroup hefftepacking
 * \brief Holds the plan for a pack/unpack operation.
 */
struct pack_plan_3d{
    //! \brief Number of elements in the three directions.
    std::array<int, 3> size;
    //! \brief Stride of the lines.
    int line_stride;
    //! \brief Stride of the planes.
    int plane_stride;
    //! \brief Stride of the lines in the received buffer (transpose packing only).
    int buff_line_stride;
    //! \brief Stride of the planes in the received buffer (transpose packing only).
    int buff_plane_stride;
    //! \brief Maps the i,j,k indexes from input to the output (transpose packing only).
    std::array<int, 3> map;
};

/*!
 * \ingroup hefftepacking
 * \brief Writes a plan to the stream, useful for debugging.
 */
inline std::ostream & operator << (std::ostream &os, pack_plan_3d const &plan){
    os << "nfast = " << plan.size[0] << "\n";
    os << "nmid  = " << plan.size[1] << "\n";
    os << "nslow = " << plan.size[2] << "\n";
    os << "line_stride = "  << plan.line_stride << "\n";
    os << "plane_stride = " << plan.plane_stride << "\n";
    if (plan.buff_line_stride > 0){
        os << "buff_line_stride = " << plan.buff_line_stride << "\n";
        os << "buff_plane_stride = " << plan.buff_plane_stride << "\n";
        os << "map = (" << plan.map[0] << ", " << plan.map[1] << ", " << plan.map[2] << ")\n";
    }
    os << "\n";
    return os;
}

/*!
 * \ingroup hefftepacking
 * \brief The packer needs to know whether the data will be on the CPU or GPU devices.
 *
 * Specializations of this template will define the type alias \b mode
 * that will be set to either the tag::cpu or tag::gpu.
 */
template<typename backend>
struct packer_backend{};

// typename struct packer_backend<cuda>{ using mode = tag::gpu; } // specialization can differentiate between gpu and cpu backends

/*!
 * \ingroup hefftepacking
 * \brief Defines the direct packer without implementation, use the specializations to get the CPU or GPU implementation.
 */
template<typename mode> struct direct_packer{};

/*!
 * \ingroup hefftepacking
 * \brief Simple packer that copies sub-boxes without transposing the order of the indexes.
 */
template<> struct direct_packer<tag::cpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        scalar_type* buffer_iterator = buffer;
        for(int slow = 0; slow < plan.size[2]; slow++){
            for(int mid = 0; mid < plan.size[1]; mid++){
                buffer_iterator = std::copy_n(&data[slow * plan.plane_stride + mid * plan.line_stride], plan.size[0], buffer_iterator);
            }
        }
    }
    //! \brief Execute the planned unpack operation.
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        for(int slow = 0; slow < plan.size[2]; slow++){
            for(int mid = 0; mid < plan.size[1]; mid++){
                std::copy_n(&buffer[(slow * plan.size[1] + mid) * plan.size[0]],
                            plan.size[0], &data[slow * plan.plane_stride + mid * plan.line_stride]);
            }
        }
    }
};

/*!
 * \ingroup hefftepacking
 * \brief Defines the transpose packer without implementation, use the specializations to get the CPU implementation.
 */
template<typename mode> struct transpose_packer{};

/*!
 * \ingroup hefftepacking
 * \brief Transpose packer that packs sub-boxes without transposing, but unpacks applying a transpose operation.
 */
template<> struct transpose_packer<tag::cpu>{
    //! \brief Execute the planned pack operation.
    template<typename scalar_type>
    void pack(pack_plan_3d const &plan, scalar_type const data[], scalar_type buffer[]) const{
        direct_packer<tag::cpu>().pack(plan, data, buffer); // packing is done the same way as the direct_packer
    }
    /*!
     * \brief Execute the planned unpack operation.
     *
     * Note that this will transpose the data in the process.
     * The transpose is done in blocks to maximize cache reuse.
     */
    template<typename scalar_type>
    void unpack(pack_plan_3d const &plan, scalar_type const buffer[], scalar_type data[]) const{
        constexpr int stride = 256 / sizeof(scalar_type);
        if (plan.map[0] == 0 and plan.map[1] == 1){
            for(int i=0; i<plan.size[2]; i++)
                for(int j=0; j<plan.size[1]; j++)
                    for(int k=0; k<plan.size[0]; k++)
                        data[i * plan.plane_stride + j * plan.line_stride + k]
                            = buffer[ i * plan.buff_plane_stride + j * plan.buff_line_stride + k ];

        }else if (plan.map[0] == 0 and plan.map[1] == 2){
            for(int bi=0; bi<plan.size[2]; bi+=stride)
                for(int bj=0; bj<plan.size[1]; bj+=stride)
                    for(int bk=0; bk<plan.size[0]; bk+=stride)
                        for(int i=bi; i<std::min(bi + stride, plan.size[2]); i++)
                            for(int j=bj; j<std::min(bj + stride, plan.size[1]); j++)
                                for(int k=bk; k<std::min(bk + stride, plan.size[0]); k++)
                                    data[i * plan.plane_stride + j * plan.line_stride + k]
                                        = buffer[ j * plan.buff_plane_stride + i * plan.buff_line_stride + k ];

        }else if (plan.map[0] == 1 and plan.map[1] == 0){
            for(int bi=0; bi<plan.size[2]; bi+=stride)
                for(int bj=0; bj<plan.size[1]; bj+=stride)
                    for(int bk=0; bk<plan.size[0]; bk+=stride)
                        for(int i=bi; i<std::min(bi + stride, plan.size[2]); i++)
                            for(int j=bj; j<std::min(bj + stride, plan.size[1]); j++)
                                for(int k=bk; k<std::min(bk + stride, plan.size[0]); k++)
                                    data[i * plan.plane_stride + j * plan.line_stride + k]
                                        = buffer[ i * plan.buff_plane_stride + k * plan.buff_line_stride + j ];

        }else if (plan.map[0] == 1 and plan.map[1] == 2){
            for(int bi=0; bi<plan.size[2]; bi+=stride)
                for(int bj=0; bj<plan.size[1]; bj+=stride)
                    for(int bk=0; bk<plan.size[0]; bk+=stride)
                        for(int i=bi; i<std::min(bi + stride, plan.size[2]); i++)
                            for(int j=bj; j<std::min(bj + stride, plan.size[1]); j++)
                                for(int k=bk; k<std::min(bk + stride, plan.size[0]); k++)
                                    data[i * plan.plane_stride + j * plan.line_stride + k]
                                        = buffer[ k * plan.buff_plane_stride + i * plan.buff_line_stride + j ];

        }else if (plan.map[0] == 2 and plan.map[1] == 0){
            for(int bi=0; bi<plan.size[2]; bi+=stride)
                for(int bj=0; bj<plan.size[1]; bj+=stride)
                    for(int bk=0; bk<plan.size[0]; bk+=stride)
                        for(int i=bi; i<std::min(bi + stride, plan.size[2]); i++)
                            for(int j=bj; j<std::min(bj + stride, plan.size[1]); j++)
                                for(int k=bk; k<std::min(bk + stride, plan.size[0]); k++)
                                    data[i * plan.plane_stride + j * plan.line_stride + k]
                                        = buffer[ j * plan.buff_plane_stride + k * plan.buff_line_stride + i ];

        }else{ // if (plan.map[0] == 2 and plan.map[1] == 1){
            for(int bi=0; bi<plan.size[2]; bi+=stride)
                for(int bj=0; bj<plan.size[1]; bj+=stride)
                    for(int bk=0; bk<plan.size[0]; bk+=stride)
                        for(int i=bi; i<std::min(bi + stride, plan.size[2]); i++)
                            for(int j=bj; j<std::min(bj + stride, plan.size[1]); j++)
                                for(int k=bk; k<std::min(bk + stride, plan.size[0]); k++)
                                    data[i * plan.plane_stride + j * plan.line_stride + k]
                                        = buffer[ k * plan.buff_plane_stride + j * plan.buff_line_stride + i ];

        }

    }
};

/*!
 * \ingroup hefftepacking
 * \brief Apply scaling to the CPU data.
 *
 * Similar to the packer, the scaling factors are divided into CPU and GPU variants
 * and not specific to the backend, e.g., FFTW and MKL use the same CPU scaling method.
 */
template<typename mode> struct data_scaling{};

/*!
 * \ingroup hefftepacking
 * \brief Specialization for the CPU case.
 */
template<> struct data_scaling<tag::cpu>{
    /*!
     * \ingroup hefftepacking
     * \brief Simply multiply the \b num_entries in the \b data by the \b scale_factor.
     */
    template<typename scalar_type>
    static void apply(int num_entries, scalar_type *data, double scale_factor){;
        for(int i=0; i<num_entries; i++) data[i] *= scale_factor;
    }
    /*!
     * \ingroup hefftepacking
     * \brief Complex by real scaling.
     *
     * Depending on the compiler and type of operation, C++ complex numbers can have bad
     * performance compared to float and double operations.
     * Since the scaling factor is always real, scaling can be performed
     * with real arithmetic which is easier to vectorize.
     */
    template<typename precision_type>
    static void apply(int num_entries, std::complex<precision_type> *data, double scale_factor){
        apply<precision_type>(2*num_entries, reinterpret_cast<precision_type*>(data), scale_factor);
    }
};

}

#endif
