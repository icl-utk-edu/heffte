/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_MKL_H
#define HEFFTE_BACKEND_MKL_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_MKL

#define FFT_MKL
#include "mkl_dfti.h"

namespace heffte{

namespace backend{
    //! \brief Type-tag for the MKL backend
    struct mkl{};

    //! \brief Indicate that the MKL backend has been enabled.
    template<> struct is_enabled<mkl> : std::true_type{};

    /*!
     * \brief Returns the human readable name of the MKL backend.
     */
    template<> inline std::string name<mkl>(){ return "mkl"; }
}

/*!
 * \brief Recognize the MKL single precision complex type.
 */
template<> struct is_ccomplex<float _Complex> : std::true_type{};
/*!
 * \brief Recognize the MKL double precision complex type.
 */
template<> struct is_zcomplex<double _Complex> : std::true_type{};

/*!
 * \brief Base plan for mkl, using only the specialization for float and double complex.
 *
 * MKL library uses a unique plan type for forward and backward fft transforms.
 * The specializations to this struct will wrap around such plan and provide RAII style
 * of memory management and simple constructors that take inputs suitable to HeFFTe.
 */

template<typename> struct plan_mkl{};

inline void check_error(MKL_LONG status, std::string const &function_name){
    if (status != 0){
        throw std::runtime_error(function_name + " failed with status: " + std::to_string(status));
    }
}

/*!
 * \brief Plan for single precision complex transform.
 */
template<>
struct plan_mkl<std::complex<float>>{
    /*!
     * \brief Constructor, takes inputs identical to MKL FFT descriptors.
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmanyffts is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_mkl(int size, int howmanyffts, int stride, int dist) : plan(nullptr){
        check_error( DftiCreateDescriptor(&plan, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG) size), "mkl plan create" );
        check_error( DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts), "mkl set howmany");
        check_error( DftiSetValue(plan, DFTI_PLACEMENT, DFTI_INPLACE), "mkl set in place");
        MKL_LONG lstride[] = {0, static_cast<MKL_LONG>(stride)};
        check_error( DftiSetValue(plan, DFTI_INPUT_STRIDES, lstride), "mkl set istride");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_STRIDES, lstride), "mkl set ostride");
        check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) dist), "mkl set idist");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) dist), "mkl set odist");
        check_error( DftiCommitDescriptor(plan), "mkl commit");
    }

    //! \brief Destructor, deletes the plan.
    ~plan_mkl(){ check_error( DftiFreeDescriptor(&plan), "mkl delete descriptor"); }
    //! \brief Custom conversion to the MKL plan.
    operator DFTI_DESCRIPTOR_HANDLE() const{ return plan; }
    //! \brief The MKL opaque structure (pointer to struct).
    DFTI_DESCRIPTOR_HANDLE plan;
};

/*!
//! \brief Specialization for double complex.
 */
template<>
struct plan_mkl<std::complex<double>>{
    /*!
     * \brief Constructor, takes inputs identical to MKL FFT descriptors.
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmanyffts is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_mkl(int size, int howmanyffts, int stride, int dist) : plan(nullptr){
        check_error( DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG) size), "mkl plan create" );
        check_error( DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts), "mkl set howmany");
        check_error( DftiSetValue(plan, DFTI_PLACEMENT, DFTI_INPLACE), "mkl set in place");
        MKL_LONG lstride[] = {0, static_cast<MKL_LONG>(stride)};
        check_error( DftiSetValue(plan, DFTI_INPUT_STRIDES, lstride), "mkl set istride");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_STRIDES, lstride), "mkl set ostride");
        check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) dist), "mkl set idist");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) dist), "mkl set odist");
        check_error( DftiCommitDescriptor(plan), "mkl commit");
    }

    //! \brief Destructor, deletes the plan.
    ~plan_mkl(){ check_error( DftiFreeDescriptor(&plan), "mkl delete descriptor"); }
    //! \brief Custom conversion to the MKL plan.
    operator DFTI_DESCRIPTOR_HANDLE() const{ return plan; }
    //! \brief The MKL opaque structure (pointer to struct).
    DFTI_DESCRIPTOR_HANDLE plan;
};



/*!
 * \brief Wrapper around the MKL API.
 *
 * A single class that manages the plans and executions of mkl
 * so that a single API is provided for all backends.
 * The executor operates on a box and performs 1-D FFTs
 * for the given dimension.
 * The class silently manages the plans and buffers needed
 * for the different types.
 * All input and output arrays must have size equal to the box.
 */
class mkl_executor{
public:
    //! \brief Constructor, specifies the box and dimension.
    mkl_executor(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == box.order[0]) ? size : 1),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        block_stride(box.osize(0) * box.osize(1)),
        total_size(box.count())
    {}

    //! \brief Forward fft, float-complex case.
    void forward(std::complex<float> data[]) const{
        make_plan(cplan);
        for(int i=0; i<blocks; i++){
            float _Complex* block_data = reinterpret_cast<float _Complex*>(data + i * block_stride);
            DftiComputeForward(*cplan, block_data);
        }
    }
    //! \brief Backward fft, float-complex case.
    void backward(std::complex<float> data[]) const{
        make_plan(cplan);
        for(int i=0; i<blocks; i++){
            float _Complex* block_data = reinterpret_cast<float _Complex*>(data + i * block_stride);
            DftiComputeBackward(*cplan, block_data);
        }
    }
    //! \brief Forward fft, double-complex case.
    void forward(std::complex<double> data[]) const{
        make_plan(zplan);
        for(int i=0; i<blocks; i++){
            double _Complex* block_data = reinterpret_cast<double _Complex*>(data + i * block_stride);
            DftiComputeForward(*zplan, block_data);
        }
    }
    //! \brief Backward fft, double-complex case.
    void backward(std::complex<double> data[]) const{
        make_plan(zplan);
        for(int i=0; i<blocks; i++){
            double _Complex* block_data = reinterpret_cast<double _Complex*>(data + i * block_stride);
            DftiComputeBackward(*zplan, block_data);
        }
    }

    //! \brief Converts the deal data to complex and performs float-complex forward transform.
    void forward(float const indata[], std::complex<float> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<float>(indata[i]);
        forward(outdata);
    }
    //! \brief Performs backward float-complex transform and truncates the complex part of the result.
    void backward(std::complex<float> indata[], float outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }
    //! \brief Converts the deal data to complex and performs double-complex forward transform.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<double>(indata[i]);
        forward(outdata);
    }
    //! \brief Performs backward double-complex transform and truncates the complex part of the result.
    void backward(std::complex<double> indata[], double outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    //! \brief Helper template to create the plan.
    template<typename scalar_type>
    void make_plan(std::unique_ptr<plan_mkl<scalar_type>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_mkl<scalar_type>>(new plan_mkl<scalar_type>(size, howmanyffts, stride, dist));
    }

    int size, howmanyffts, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_mkl<std::complex<float>>> cplan;
    mutable std::unique_ptr<plan_mkl<std::complex<double>>> zplan;
};

/*!
 * \brief Unlike the C2C plan R2C is non-symmetric and it requires that the direction is build into the plan.
 */
template<typename, direction> struct plan_mkl_r2c{};

//! \brief Specialization for r2c single precision.
template<direction dir>
struct plan_mkl_r2c<float, dir>{
    /*!
     * \brief Constructor taking into account the different sizes for the real and complex parts.
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmanyffts is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     */
    plan_mkl_r2c(int size, int howmanyffts, int stride, int rdist, int cdist) : plan(nullptr){
        check_error( DftiCreateDescriptor(&plan, DFTI_SINGLE, DFTI_REAL, 1, (MKL_LONG) size), "mkl create r2c");
        check_error( DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts), "mkl set howmany r2c");
        check_error( DftiSetValue(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "mkl set not in place r2c");
        check_error( DftiSetValue(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "mkl conj storage cc");
        MKL_LONG lstride[] = {0, static_cast<MKL_LONG>(stride)};
        check_error( DftiSetValue(plan, DFTI_INPUT_STRIDES, lstride), "mkl set istride r2c");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_STRIDES, lstride), "mkl set ostride r2c");
        if (dir == direction::forward){
            check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) rdist), "mkl set rdist r2c");
            check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) cdist), "mkl set cdist r2c");
        }else{
            check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) rdist), "mkl set back rdist r2c");
            check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) cdist), "mkl set back cdist r2c");
        }
        check_error( DftiCommitDescriptor(plan), "mkl commit r2c");
    }


    //! \brief Identical to the float-complex specialization.
    ~plan_mkl_r2c(){ check_error( DftiFreeDescriptor(&plan), "mkl free r2c"); }
    //! \brief Identical to the float-complex specialization.
    operator DFTI_DESCRIPTOR_HANDLE() const{ return plan; }
    //! \brief Identical to the float-complex specialization.
    DFTI_DESCRIPTOR_HANDLE plan;
};

//! \brief Specialization for r2c double precision.
template<direction dir>
struct plan_mkl_r2c<double, dir>{
    //! \brief Identical to the float-complex specialization.
    plan_mkl_r2c(int size, int howmanyffts, int stride, int rdist, int cdist) : plan(nullptr){
        check_error( DftiCreateDescriptor(&plan, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG) size), "mkl create r2c");
        check_error( DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG) howmanyffts), "mkl set howmany r2c");
        check_error( DftiSetValue(plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE), "mkl set not in place r2c");
        check_error( DftiSetValue(plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "mkl conj storage cc");
        MKL_LONG lstride[] = {0, static_cast<MKL_LONG>(stride)};
        check_error( DftiSetValue(plan, DFTI_INPUT_STRIDES, lstride), "mkl set istride r2c");
        check_error( DftiSetValue(plan, DFTI_OUTPUT_STRIDES, lstride), "mkl set ostride r2c");
        if (dir == direction::forward){
            check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) rdist), "mkl set rdist r2c");
            check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) cdist), "mkl set cdist r2c");
        }else{
            check_error( DftiSetValue(plan, DFTI_OUTPUT_DISTANCE, (MKL_LONG) rdist), "mkl set back rdist r2c");
            check_error( DftiSetValue(plan, DFTI_INPUT_DISTANCE, (MKL_LONG) cdist), "mkl set back cdist r2c");
        }
        check_error( DftiCommitDescriptor(plan), "mkl commit r2c");
    }
    //! \brief Identical to the float-complex specialization.
    ~plan_mkl_r2c(){ check_error( DftiFreeDescriptor(&plan), "mkl free r2c"); }
    //! \brief Identical to the float-complex specialization.
    operator DFTI_DESCRIPTOR_HANDLE() const{ return plan; }
    //! \brief Identical to the float-complex specialization.
    DFTI_DESCRIPTOR_HANDLE plan;
};

/*!
 * \brief Wrapper to mkl API for real-to-complex transform with shortening of the data.
 *
 * Serves the same purpose of heffte::mkl_executor but only real input is accepted
 * and only the unique (non-conjugate) coefficients are computed.
 * All real arrays must have size of real_size() and all complex arrays must have size complex_size().
 */
class mkl_executor_r2c{
public:
    /*!
     * \brief Constructor defines the box and the dimension of reduction.
     *
     * Note that the result sits in the box returned by box.r2c(dimension).
     */
    mkl_executor_r2c(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmanyffts(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        blocks((dimension == box.order[1]) ? box.osize(2) : 1),
        rdist((dimension == box.order[0]) ? size : 1),
        cdist((dimension == box.order[0]) ? size/2 + 1 : 1),
        rblock_stride(box.osize(0) * box.osize(1)),
        cblock_stride(box.osize(0) * (box.osize(1)/2 + 1)),
        rsize(box.count()),
        csize(box.r2c(dimension).count())
    {}

    //! \brief Forward transform, single precision.
    void forward(float const indata[], std::complex<float> outdata[]) const{
        make_plan(sforward);
        for(int i=0; i<blocks; i++){
            float *rdata = const_cast<float*>(indata + i * rblock_stride);
            float _Complex* cdata = reinterpret_cast<float _Complex*>(outdata + i * cblock_stride);
            DftiComputeForward(*sforward, rdata, cdata);
        }
    }
    //! \brief Backward transform, single precision.
    void backward(std::complex<float> const indata[], float outdata[]) const{
        make_plan(sbackward);
        for(int i=0; i<blocks; i++){
            float _Complex* cdata = const_cast<float _Complex*>(reinterpret_cast<float _Complex const*>(indata + i * cblock_stride));
            DftiComputeBackward(*sbackward, cdata, outdata + i * rblock_stride);
        }
    }
    //! \brief Forward transform, double precision.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        make_plan(dforward);
        for(int i=0; i<blocks; i++){
            double *rdata = const_cast<double*>(indata + i * rblock_stride);
            double _Complex* cdata = reinterpret_cast<double _Complex*>(outdata + i * cblock_stride);
            DftiComputeForward(*dforward, rdata, cdata);
        }
    }
    //! \brief Backward transform, double precision.
    void backward(std::complex<double> const indata[], double outdata[]) const{
        make_plan(dbackward);
        for(int i=0; i<blocks; i++){
            double _Complex* cdata = const_cast<double _Complex*>(reinterpret_cast<double _Complex const*>(indata + i * cblock_stride));
            DftiComputeBackward(*dbackward, cdata, outdata + i * rblock_stride);
        }
    }

    //! \brief Returns the size of the box with real data.
    int real_size() const{ return rsize; }
    //! \brief Returns the size of the box with complex coefficients.
    int complex_size() const{ return csize; }

private:
    //! \brief Helper template to initialize the plan.
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_mkl_r2c<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_mkl_r2c<scalar_type, dir>>(new plan_mkl_r2c<scalar_type, dir>(size, howmanyffts, stride, rdist, cdist));
    }

    int size, howmanyffts, stride, blocks;
    int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_mkl_r2c<float, direction::forward>> sforward;
    mutable std::unique_ptr<plan_mkl_r2c<double, direction::forward>> dforward;
    mutable std::unique_ptr<plan_mkl_r2c<float, direction::backward>> sbackward;
    mutable std::unique_ptr<plan_mkl_r2c<double, direction::backward>> dbackward;
};

/*!
 * \brief Helper struct that defines the types and creates instances of one-dimensional executors.
 *
 * The struct is specialized for each backend.
 */
template<> struct one_dim_backend<backend::mkl>{
    //! \brief Defines the complex-to-complex executor.
    using type = mkl_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = mkl_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    static std::unique_ptr<mkl_executor> make(box3d const box, int dimension){
        return std::unique_ptr<mkl_executor>(new mkl_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    static std::unique_ptr<mkl_executor_r2c> make_r2c(box3d const box, int dimension){
        return std::unique_ptr<mkl_executor_r2c>(new mkl_executor_r2c(box, dimension));
    }
};

/*!
 * \brief Sets the default options for the mkl backend.
 */
template<> struct default_plan_options<backend::mkl>{
    //! \brief The reshape operations will not transpose the data.
    static const bool use_reorder = false;
};

}

#endif

#endif   /* HEFFTE_BACKEND_MKL_H */
