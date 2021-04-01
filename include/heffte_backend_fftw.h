/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_FFTW_H
#define HEFFTE_BACKEND_FFTW_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_FFTW

#include "fftw3.h"

/*!
 * \ingroup fft3d
 * \addtogroup hefftefftw Backend fftw3
 *
 * Wrappers and template specializations related to the FFTW backend.
 * Requires CMake option:
 * \code
 *  -D Heffte_ENABLE_FFTW=ON
 * \endcode
 */

namespace heffte{

namespace backend{
    /*!
     * \ingroup hefftefftw
     * \brief Type-tag for the FFTW backend
     */
    struct fftw{};

    /*!
     * \ingroup hefftefftw
     * \brief Indicate that the FFTW backend has been enabled.
     */
    template<> struct is_enabled<fftw> : std::true_type{};

// Specialization is not necessary since the default behavior assumes CPU parameters.
//     template<>
//     struct buffer_traits<fftw>{
//         using location = tag::cpu;
//         template<typename T> using container = std::vector<T>;
//     };

    /*!
     * \ingroup hefftefftw
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<fftw>(){ return "fftw"; }
}

/*!
 * \ingroup hefftefftw
 * \brief Recognize the FFTW single precision complex type.
 */
template<> struct is_ccomplex<fftwf_complex> : std::true_type{};
/*!
 * \ingroup hefftefftw
 * \brief Recognize the FFTW double precision complex type.
 */
template<> struct is_zcomplex<fftw_complex> : std::true_type{};

/*!
 * \ingroup hefftefftw
 * \brief Base plan for fftw, using only the specialization for float and double complex.
 *
 * FFTW3 library uses plans for forward and backward fft transforms.
 * The specializations to this struct will wrap around such plans and provide RAII style
 * of memory management and simple constructors that take inputs suitable to HeFFTe.
 */
template<typename, direction> struct plan_fftw{};

/*!
 * \ingroup hefftefftw
 * \brief Plan for the single precision complex transform.
 *
 * \tparam dir indicates a forward or backward transform
 */
template<direction dir>
struct plan_fftw<std::complex<float>, dir>{
    /*!
     * \brief Constructor, takes inputs identical to fftwf_plan_many_dft().
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmanyffts is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_fftw(int size, int howmanyffts, int stride, int dist) :
        plan(fftwf_plan_many_dft(1, &size, howmanyffts, nullptr, nullptr, stride, dist,
                                                    nullptr, nullptr, stride, dist,
                                                    (dir == direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE
                                ))
        {}
    //! \brief Destructor, deletes the plan.
    ~plan_fftw(){ fftwf_destroy_plan(plan); }
    //! \brief Custom conversion to the FFTW3 plan.
    operator fftwf_plan() const{ return plan; }
    //! \brief The FFTW3 opaque structure (pointer to struct).
    fftwf_plan plan;
};
/*!
 * \ingroup hefftefftw
 * \brief Specialization for double complex.
 */
template<direction dir>
struct plan_fftw<std::complex<double>, dir>{
    //! \brief Identical to the float-complex specialization.
    plan_fftw(int size, int howmanyffts, int stride, int dist) :
        plan(fftw_plan_many_dft(1, &size, howmanyffts, nullptr, nullptr, stride, dist,
                                                   nullptr, nullptr, stride, dist,
                                                   (dir == direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE
                               ))
        {}
    //! \brief Identical to the float-complex specialization.
    ~plan_fftw(){ fftw_destroy_plan(plan); }
    //! \brief Identical to the float-complex specialization.
    operator fftw_plan() const{ return plan; }
    //! \brief Identical to the float-complex specialization.
    fftw_plan plan;
};

/*!
 * \ingroup hefftefftw
 * \brief Wrapper around the FFTW3 API.
 *
 * A single class that manages the plans and executions of fftw3
 * so that a single API is provided for all backends.
 * The executor operates on a box and performs 1-D FFTs
 * for the given dimension.
 * The class silently manages the plans and buffers needed
 * for the different types.
 * All input and output arrays must have size equal to the box.
 */
class fftw_executor{
public:
    //! \brief Constructor, specifies the box and dimension.
    template<typename index>
    fftw_executor(box3d<index> const box, int dimension) :
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
        make_plan(cforward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
            fftwf_execute_dft(*cforward, block_data, block_data);
        }
    }
    //! \brief Backward fft, float-complex case.
    void backward(std::complex<float> data[]) const{
        make_plan(cbackward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
            fftwf_execute_dft(*cbackward, block_data, block_data);
        }
    }
    //! \brief Forward fft, double-complex case.
    void forward(std::complex<double> data[]) const{
        make_plan(zforward);
        for(int i=0; i<blocks; i++){
            fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
            fftw_execute_dft(*zforward, block_data, block_data);
        }
    }
    //! \brief Backward fft, double-complex case.
    void backward(std::complex<double> data[]) const{
        make_plan(zbackward);
        for(int i=0; i<blocks; i++){
            fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
            fftw_execute_dft(*zbackward, block_data, block_data);
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
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_fftw<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_fftw<scalar_type, dir>>(new plan_fftw<scalar_type, dir>(size, howmanyffts, stride, dist));
    }

    int size, howmanyffts, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::forward>> cforward;
    mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::backward>> cbackward;
    mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::forward>> zforward;
    mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::backward>> zbackward;
};

/*!
 * \ingroup hefftefftw
 * \brief Specialization for r2c single precision.
 */
template<direction dir>
struct plan_fftw<float, dir>{
    /*!
     * \brief Constructor taking into account the different sizes for the real and complex parts.
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmanyffts is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param rdist is the distance between the first entries of consecutive sequences in the real sequences
     * \param cdist is the distance between the first entries of consecutive sequences in the complex sequences
     */
    plan_fftw(int size, int howmanyffts, int stride, int rdist, int cdist) :
        plan((dir == direction::forward) ?
             fftwf_plan_many_dft_r2c(1, &size, howmanyffts, nullptr, nullptr, stride, rdist,
                                                   nullptr, nullptr, stride, cdist,
                                                   FFTW_ESTIMATE
                                   ) :
             fftwf_plan_many_dft_c2r(1, &size, howmanyffts, nullptr, nullptr, stride, cdist,
                                                   nullptr, nullptr, stride, rdist,
                                                   FFTW_ESTIMATE
                                   ))
        {}
    //! \brief Identical to the float-complex specialization.
    ~plan_fftw(){ fftwf_destroy_plan(plan); }
    //! \brief Identical to the float-complex specialization.
    operator fftwf_plan() const{ return plan; }
    //! \brief Identical to the float-complex specialization.
    fftwf_plan plan;
};
/*!
 * \ingroup hefftefftw
 * \brief Specialization for r2c double precision.
 */
template<direction dir>
struct plan_fftw<double, dir>{
    //! \brief Identical to the float-complex specialization.
    plan_fftw(int size, int howmanyffts, int stride, int rdist, int cdist) :
        plan((dir == direction::forward) ?
             fftw_plan_many_dft_r2c(1, &size, howmanyffts, nullptr, nullptr, stride, rdist,
                                                   nullptr, nullptr, stride, cdist,
                                                   FFTW_ESTIMATE
                                   ) :
             fftw_plan_many_dft_c2r(1, &size, howmanyffts, nullptr, nullptr, stride, cdist,
                                                   nullptr, nullptr, stride, rdist,
                                                   FFTW_ESTIMATE
                                   ))
        {}
    //! \brief Identical to the float-complex specialization.
    ~plan_fftw(){ fftw_destroy_plan(plan); }
    //! \brief Identical to the float-complex specialization.
    operator fftw_plan() const{ return plan; }
    //! \brief Identical to the float-complex specialization.
    fftw_plan plan;
};

/*!
 * \ingroup hefftefftw
 * \brief Wrapper to fftw3 API for real-to-complex transform with shortening of the data.
 *
 * Serves the same purpose of heffte::fftw_executor but only real input is accepted
 * and only the unique (non-conjugate) coefficients are computed.
 * All real arrays must have size of real_size() and all complex arrays must have size complex_size().
 */
class fftw_executor_r2c{
public:
    /*!
     * \brief Constructor defines the box and the dimension of reduction.
     *
     * Note that the result sits in the box returned by box.r2c(dimension).
     */
    template<typename index>
    fftw_executor_r2c(box3d<index> const box, int dimension) :
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
            fftwf_complex* cdata = reinterpret_cast<fftwf_complex*>(outdata + i * cblock_stride);
            fftwf_execute_dft_r2c(*sforward, rdata, cdata);
        }
    }
    //! \brief Backward transform, single precision.
    void backward(std::complex<float> const indata[], float outdata[]) const{
        make_plan(sbackward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* cdata = const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex const*>(indata + i * cblock_stride));
            fftwf_execute_dft_c2r(*sbackward, cdata, outdata + i * rblock_stride);
        }
    }
    //! \brief Forward transform, double precision.
    void forward(double const indata[], std::complex<double> outdata[]) const{
        make_plan(dforward);
        for(int i=0; i<blocks; i++){
            double *rdata = const_cast<double*>(indata + i * rblock_stride);
            fftw_complex* cdata = reinterpret_cast<fftw_complex*>(outdata + i * cblock_stride);
            fftw_execute_dft_r2c(*dforward, rdata, cdata);
        }
    }
    //! \brief Backward transform, double precision.
    void backward(std::complex<double> const indata[], double outdata[]) const{
        make_plan(dbackward);
        for(int i=0; i<blocks; i++){
            fftw_complex* cdata = const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(indata + i * cblock_stride));
            fftw_execute_dft_c2r(*dbackward, cdata, outdata + i * rblock_stride);
        }
    }

    //! \brief Returns the size of the box with real data.
    int real_size() const{ return rsize; }
    //! \brief Returns the size of the box with complex coefficients.
    int complex_size() const{ return csize; }

private:
    //! \brief Helper template to initialize the plan.
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_fftw<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_fftw<scalar_type, dir>>(new plan_fftw<scalar_type, dir>(size, howmanyffts, stride, rdist, cdist));
    }

    int size, howmanyffts, stride, blocks;
    int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_fftw<float, direction::forward>> sforward;
    mutable std::unique_ptr<plan_fftw<double, direction::forward>> dforward;
    mutable std::unique_ptr<plan_fftw<float, direction::backward>> sbackward;
    mutable std::unique_ptr<plan_fftw<double, direction::backward>> dbackward;
};

/*!
 * \ingroup hefftefftw
 * \brief Helper struct that defines the types and creates instances of one-dimensional executors.
 *
 * The struct is specialized for each backend.
 */
template<> struct one_dim_backend<backend::fftw>{
    //! \brief Defines the complex-to-complex executor.
    using type = fftw_executor;
    //! \brief Defines the real-to-complex executor.
    using type_r2c = fftw_executor_r2c;

    //! \brief Constructs a complex-to-complex executor.
    template<typename index>
    static std::unique_ptr<fftw_executor> make(void*, box3d<index> const box, int dimension){
        return std::unique_ptr<fftw_executor>(new fftw_executor(box, dimension));
    }
    //! \brief Constructs a real-to-complex executor.
    template<typename index>
    static std::unique_ptr<fftw_executor_r2c> make_r2c(void*, box3d<index> const box, int dimension){
        return std::unique_ptr<fftw_executor_r2c>(new fftw_executor_r2c(box, dimension));
    }
};

/*!
 * \ingroup hefftefftw
 * \brief Sets the default options for the fftw backend.
 */
template<> struct default_plan_options<backend::fftw>{
    //! \brief The reshape operations will also reorder the data.
    static const bool use_reorder = true;
};

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
