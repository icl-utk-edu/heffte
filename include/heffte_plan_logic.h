/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_PLAN_LOGIC_H
#define HEFFTE_PLAN_LOGIC_H

#include "heffte_common.h"

/*!
 * \ingroup fft3d
 * \addtogroup fft3dplan Plan transformation logic
 *
 * Implements the analysis of the input and output distribution of the boxes
 * and creates the corresponding plan for reshape and 1-D FFT operations.
 */

namespace heffte {

/*!
 * \ingroup fft3d
 * \brief Defines a set of tweaks and options to use in the plan generation.
 *
 * Example usage:
 * \code
 *  heffte::plan_options options = heffte::default_options<heffte::backend::fftw>();
 *  options.use_alltoall = false; // forces the use of point-to-point communication
 *  heffte::fft3d<heffte::backend::fftw> fft3d(inbox, outbox, comm, options);
 * \endcode
 */
struct plan_options{
    //! \brief Constructor, initializes all options with the default values for the given backend tag.
    template<typename backend_tag> plan_options(backend_tag const)
        : use_reorder(default_plan_options<backend_tag>::use_reorder),
          use_alltoall(true),
          use_pencils(true),
          use_gpu_aware(true)
    {}
    //! \brief Constructor, initializes each variable, primarily for internal use.
    plan_options(bool reorder, bool alltoall, bool pencils)
        : use_reorder(reorder), use_alltoall(alltoall), use_pencils(pencils)
    {}
    //! \brief Defines whether to transpose the data on reshape or to use strided 1-D ffts.
    bool use_reorder;
    //! \brief Defines whether to use point to point or all to all communications.
    bool use_alltoall;
    //! \brief Defines whether to use pencil or slab data distribution in the reshape steps.
    bool use_pencils;
    //! \brief Defines whether to use MPI calls directly from the GPU or to move to the CPU first.
    bool use_gpu_aware;
};

/*!
 * \ingroup fft3d
 * \brief Simple I/O for the plan options struct.
 */
inline std::ostream & operator << (std::ostream &os, plan_options const options){
    os << "options = ("
       << ((options.use_reorder) ? "fft1d:contiguous" : "fft1d:strided") << ", "
       << ((options.use_alltoall) ? "mpi:alltoallv" : "mpi:point-to-point") << ", "
       << ((options.use_pencils) ? "decomposition:pencil" : "decomposition:slab") << ", "
       << ((options.use_gpu_aware) ? "mpi:from-gpu" : "mpi:from-cpu") << ")";
    return os;
}

/*!
 * \ingroup heffterocm
 * \brief Forces the reorder logic for the ROCM r2c variant.
 */
inline plan_options force_reorder(plan_options opts){
    opts.use_reorder = true;
    return opts;
}

/*!
 * \ingroup fft3d
 * \brief Returns the default backend options associated with the given backend.
 */
template<typename backend_tag>
plan_options default_options(){
    return plan_options(backend_tag());
}

/*!
 * \ingroup fft3dplan
 * \brief The logic plan incorporates the order and types of operations in a transform.
 *
 * The logic_plan is used to separate the logic of the order of basic operations (reshape or fft execute)
 * from the constructor of the fft3d and fft3d_r2c classes.
 * In this manner, detection of pencils vs. brick distribution of the data and/or making decisions regarding
 * the transposition of indexing can be done in sufficiently robust and complex logic without
 * clutter of the main classes or unnecessary repetition of code.
 *
 * Node that a reshape operation \b i will be performed only if in_shape[i] and out_shape[i] are different.
 *
 * Specifically:
 * - in_shape[0] is the set of input boxes
 * - out_shape[0] is the geometry to be used for the fist 1-D FFT operation which will happen in fft_direction[0]
 * - in_shape[1] is either out_shape[0] or, in the r2c case, the boxes with reduced dimension
 * - in_shape[i] -> out_shape[i] for i = 1, 2, will hold the intermediate shapes
 * - in_shape[i] may be equal to out_shape[i] for any i = 0, 1, 2, 3
 * - 1-D FFT transforms will be applied to out_shape[1] and out_shape[2] in directions fft_direction[0] and fft_direction[1]
 * - out_shape[3] is the set of output boxes
 * - index_count is the produce of all indexes, used in scaling
 * - options will hold a copy of the set of options used in the construction
 */
template<typename index>
struct logic_plan3d{
    //! \brief Holds the input shapes for the 4 forward reshapes (backwards reverses in and out).
    std::vector<box3d<index>> in_shape[4];
    //! \brief Holds the output shapes for the 4 forward reshapes (backwards reverses in and out).
    std::vector<box3d<index>> out_shape[4];
    //! \brief Direction of the 1-D FFT transforms.
    std::array<int, 3> fft_direction;
    //! \brief The total number of indexes in all directions.
    long long index_count;
    //! \brief Extra options used in the plan creation.
    plan_options const options;
};

/*!
 * \ingroup fft3dplan
 * \brief Returns true for each direction where the boxes form pencils (i.e., where the size matches the world size).
 */
template<typename index>
inline std::array<bool, 3> pencil_directions(box3d<index> const world, std::vector<box3d<index>> const &boxes){
    std::array<bool, 3> is_pencil = {true, true, true};
    for(auto const &b : boxes){
        for(int i=0; i<3; i++)
            is_pencil[i] = is_pencil[i] and (world.size[i] == b.size[i]);
    }
    return is_pencil;
}

/*!
 * \ingroup fft3dplan
 * \brief Creates the logic plan with the provided user input.
 *
 * \param boxes is the current distribution of the data across the MPI comm
 * \param r2c_direction is the direction is the direction of shrinking of the data for an r2c transform
 *              the c2c case should use -1
 * \param options is a set of heffte::plan_options to use
 *
 * \returns the plan for reshape and 1-D fft transformations
 */
template<typename index>
logic_plan3d<index> plan_operations(ioboxes<index> const &boxes, int r2c_direction, plan_options const options);

}

#endif
