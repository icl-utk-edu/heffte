/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_C_DEFINES_H
#define HEFFTE_C_DEFINES_H

/*!
 * \defgroup cfft C Binding
 *
 * \par HeFFTe C Wrappers
 * Encapsulates a set of wrappers that provide functional C API.
 * The wrappers are included through the same header "heffte.h"
 * but with a C compiler (i.e., missing __cplusplus macro).
 *
 * The templated classes heffte::fft3d and heffte::fft3d_r2c are replaced by opaque plan handles of type heffte_plan.
 * The plan has to be created and later destroyed with
 * - heffte_plan_create()
 * - heffte_plan_create_r2c()
 * - heffte_plan_destroy()
 *
 * Queries of the problem size, backend and r2c reduction of indexes can be performed with:
 * - heffte_size_inbox()
 * - heffte_size_outbox()
 * - heffte_size_workspace()
 * - heffte_get_backend()
 * - heffte_is_r2c()
 *
 * Forward and backward transforms are performed using variants of:
 * - heffte_forward_s2c()
 * - heffte_forward_s2c_buffered()
 * - heffte_backward_s2c()
 * - heffte_backward_s2c_buffered()
 *
 * The buffered versions of the methods accept an extra parameter that is a user allocated workspace.
 *
 * Complex arrays are accepted as void-pointers which removes type safety but works around the issues with multiple complex types.
 * The methods use different suffixes to indicate the input/output types using standard BLAS naming conventions:
 * - \b s for single precision real
 * - \b c for single precision complex
 * - \b d for double precision real
 * - \b z for double precision complex
 */

#include "heffte_config.h"

#ifdef Heffte_ENABLE_FFTW
/*!
 * \ingroup cfft
 * \brief Set the use of the FFTW backend.
 */
#define Heffte_BACKEND_FFTW 1
#endif
#ifdef Heffte_ENABLE_MKL
/*!
 * \ingroup cfft
 * \brief Set the use of the MKL backend.
 */
#define Heffte_BACKEND_MKL 2
#endif
#ifdef Heffte_ENABLE_CUDA
/*!
 * \ingroup cfft
 * \brief Set the use of the cuFFT backend.
 */
#define Heffte_BACKEND_CUFFT 10
#endif

/*!
 * \ingroup cfft
 * \brief The success return of a HeFFTe method.
 */
#define Heffte_SUCCESS 0

/*!
 * \ingroup cfft
 * \brief Equivalent to heffte::plan_options but defined for the C API.
 *
 * The int variables use the C logic 0 means false, not 0 means true.
 */
typedef struct{
    //! \brief Corresponds to heffte::plan_options::use_reorder
    int use_reorder;
    //! \brief Corresponds to heffte::plan_options::use_alltoall
    int use_alltoall;
    //! \brief Corresponds to heffte::plan_options::use_pencils
    int use_pencils;
} heffte_plan_options;

/*!
 * \ingroup cfft
 * \brief Generic wrapper around some fft3d/fft3d_r2c instance, use heffte_plan instead of this.
 *
 * Do not directly modify the entries of the struct, use the create and destroy methods.
 * - heffte_plan_create()
 * - heffte_plan_create_r2c()
 * - heffte_plan_destroy()
 */
typedef struct{
    //! \brief Remembers the type of the backend.
    int backend_type;
    //! \brief If 0 then using heffte::fft3d, otherwise using heffte::fft3d_r2c
    int using_r2c;
    //! \brief Wrapper around an object of type heffte::fft3d or heffte::fft3d_r2c
    void *fft;
} heffte_fft_plan;

/*!
 * \ingroup cfft
 * \brief C-style wrapper around an instance of heffte::fft3d or heffte::fft3d_r2c using some backend.
 */
typedef heffte_fft_plan* heffte_plan;

/*!
 * \ingroup cfft
 * \brief Indicate no scaling, see heffte::scale::none
 */
#define Heffte_SCALE_NONE 0
/*!
 * \ingroup cfft
 * \brief Indicate full scaling, see heffte::scale::full
 */
#define Heffte_SCALE_FULL 1
/*!
 * \ingroup cfft
 * \brief Indicate symmetric scaling, see heffte::scale::symmetric
 */
#define Heffte_SCALE_SYMMETRIC 2

#endif // HEFFTE_C_DEFINES_H
