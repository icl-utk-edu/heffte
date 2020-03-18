/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_BACKEND_FFTW_H
#define HEFFTE_BACKEND_FFTW_H

#include "heffte_pack3d.h"

#ifdef Heffte_ENABLE_FFTW

#define FFT_FFTW3
#include "fftw3.h"

namespace heffte{

namespace backend{
    //! \brief Type-tag for the FFTW backend
    struct fftw{};

    //! \brief Indicate that the FFTW backend has been enabled.
    template<> struct is_enabled<fftw> : std::true_type{};

// Specialization is not necessary since the default behavior assumes CPU parameters.
//     template<>
//     struct buffer_traits<fftw>{
//         using location = tag::cpu;
//         template<typename T> using container = std::vector<T>;
//     };

    /*!
     * \brief Returns the human readable name of the FFTW backend.
     */
    template<> inline std::string name<fftw>(){ return "fftw"; }
}

/*!
 * \brief Recognize the FFTW single precision complex type.
 */
template<> struct is_ccomplex<fftwf_complex> : std::true_type{};
/*!
 * \brief Recognize the FFTW double precision complex type.
 */
template<> struct is_zcomplex<fftw_complex> : std::true_type{};

/*!
 * \brief Base plan for fftw, using only the specialization for float and double complex.
 *
 * FFTW3 library uses plans for forward and backward fft transforms.
 * The specializations to this struct will wrap around such plans and provide RAII style
 * of memory management and simple constructors that take inputs suitable to HeFFTe.
 */
template<typename, direction> struct plan_fftw{};

/*!
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
     * \param howmany is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param dist is the distance between the first entries of consecutive sequences
     */
    plan_fftw(int size, int howmany, int stride, int dist) :
        plan(fftwf_plan_many_dft(1, &size, howmany, nullptr, nullptr, stride, dist,
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
//! \brief Specialization for double complex.
template<direction dir>
struct plan_fftw<std::complex<double>, dir>{
    //! \brief Identical to the float-complex specialization.
    plan_fftw(int size, int howmany, int stride, int dist) :
        plan(fftw_plan_many_dft(1, &size, howmany, nullptr, nullptr, stride, dist,
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

class fftw_executor{
public:
    fftw_executor(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmany(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        dist((dimension == 0) ? size : 1),
        blocks((dimension == 1) ? box.size[2] : 1),
        block_stride(box.size[0] * box.size[1]),
        total_size(box.count())
    {}

    void forward(std::complex<float> data[]) const{
        make_plan(cforward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
            fftwf_execute_dft(*cforward, block_data, block_data);
        }
    }
    void backward(std::complex<float> data[]) const{
        make_plan(cbackward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* block_data = reinterpret_cast<fftwf_complex*>(data + i * block_stride);
            fftwf_execute_dft(*cbackward, block_data, block_data);
        }
    }
    void forward(std::complex<double> data[]) const{
        make_plan(zforward);
        for(int i=0; i<blocks; i++){
            fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
            fftw_execute_dft(*zforward, block_data, block_data);
        }
    }
    void backward(std::complex<double> data[]) const{
        make_plan(zbackward);
        for(int i=0; i<blocks; i++){
            fftw_complex* block_data = reinterpret_cast<fftw_complex*>(data + i * block_stride);
            fftw_execute_dft(*zbackward, block_data, block_data);
        }
    }

    void forward(float const indata[], std::complex<float> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<float>(indata[i]);
        forward(outdata);
    }
    void backward(std::complex<float> indata[], float outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }
    void forward(double const indata[], std::complex<double> outdata[]) const{
        for(int i=0; i<total_size; i++) outdata[i] = std::complex<double>(indata[i]);
        forward(outdata);
    }
    void backward(std::complex<double> indata[], double outdata[]) const{
        backward(indata);
        for(int i=0; i<total_size; i++) outdata[i] = std::real(indata[i]);
    }

    int box_size() const{ return total_size; }

private:
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_fftw<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_fftw<scalar_type, dir>>(new plan_fftw<scalar_type, dir>(size, howmany, stride, dist));
    }

    mutable int size, howmany, stride, dist, blocks, block_stride, total_size;
    mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::forward>> cforward;
    mutable std::unique_ptr<plan_fftw<std::complex<float>, direction::backward>> cbackward;
    mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::forward>> zforward;
    mutable std::unique_ptr<plan_fftw<std::complex<double>, direction::backward>> zbackward;
};

//! \brief Specialization for r2c single precision.
template<direction dir>
struct plan_fftw<float, dir>{
    /*!
     * \brief Constructor taking into account the different sizes for the real and complex parts.
     *
     * \param size is the number of entries in a 1-D transform
     * \param howmany is the number of transforms in the batch
     * \param stride is the distance between entries of the same transform
     * \param rdist is the distance between the first entries of consecutive sequences in the real sequences
     * \param cdist is the distance between the first entries of consecutive sequences in the complex sequences
     */
    plan_fftw(int size, int howmany, int stride, int rdist, int cdist) :
        plan((dir == direction::forward) ?
             fftwf_plan_many_dft_r2c(1, &size, howmany, nullptr, nullptr, stride, rdist,
                                                   nullptr, nullptr, stride, cdist,
                                                   FFTW_ESTIMATE
                                   ) :
             fftwf_plan_many_dft_c2r(1, &size, howmany, nullptr, nullptr, stride, cdist,
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
//! \brief Specialization for r2c double precision.
template<direction dir>
struct plan_fftw<double, dir>{
    //! \brief Identical to the float-complex specialization.
    plan_fftw(int size, int howmany, int stride, int rdist, int cdist) :
        plan((dir == direction::forward) ?
             fftw_plan_many_dft_r2c(1, &size, howmany, nullptr, nullptr, stride, rdist,
                                                   nullptr, nullptr, stride, cdist,
                                                   FFTW_ESTIMATE
                                   ) :
             fftw_plan_many_dft_c2r(1, &size, howmany, nullptr, nullptr, stride, cdist,
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

class fftw_executor_r2c{
public:
    fftw_executor_r2c(box3d const box, int dimension) :
        size(box.size[dimension]),
        howmany(fft1d_get_howmany(box, dimension)),
        stride(fft1d_get_stride(box, dimension)),
        blocks((dimension == 1) ? box.size[2] : 1),
        rdist((dimension == 0) ? size : 1),
        cdist((dimension == 0) ? size/2 + 1 : 1),
        rblock_stride(box.size[0] * box.size[1]),
        cblock_stride(box.size[0] * (box.size[1]/2 + 1)),
        rsize(box.count()),
        csize(box.r2c(dimension).count())
    {}

    void forward(float const indata[], std::complex<float> outdata[]) const{
        make_plan(sforward);
        for(int i=0; i<blocks; i++){
            float *rdata = const_cast<float*>(indata + i * rblock_stride);
            fftwf_complex* cdata = reinterpret_cast<fftwf_complex*>(outdata + i * cblock_stride);
            fftwf_execute_dft_r2c(*sforward, rdata, cdata);
        }
    }
    void backward(std::complex<float> const indata[], float outdata[]) const{
        make_plan(sbackward);
        for(int i=0; i<blocks; i++){
            fftwf_complex* cdata = const_cast<fftwf_complex*>(reinterpret_cast<fftwf_complex const*>(indata + i * cblock_stride));
            fftwf_execute_dft_c2r(*sbackward, cdata, outdata + i * rblock_stride);
        }
    }
    void forward(double const indata[], std::complex<double> outdata[]) const{
        make_plan(dforward);
        for(int i=0; i<blocks; i++){
            double *rdata = const_cast<double*>(indata + i * rblock_stride);
            fftw_complex* cdata = reinterpret_cast<fftw_complex*>(outdata + i * cblock_stride);
            fftw_execute_dft_r2c(*dforward, rdata, cdata);
        }
    }
    void backward(std::complex<double> const indata[], double outdata[]) const{
        make_plan(dbackward);
        for(int i=0; i<blocks; i++){
            fftw_complex* cdata = const_cast<fftw_complex*>(reinterpret_cast<fftw_complex const*>(indata + i * cblock_stride));
            fftw_execute_dft_c2r(*dbackward, cdata, outdata + i * rblock_stride);
        }
    }

    int real_size() const{ return rsize; }
    int complex_size() const{ return csize; }

private:
    template<typename scalar_type, direction dir>
    void make_plan(std::unique_ptr<plan_fftw<scalar_type, dir>> &plan) const{
        if (!plan) plan = std::unique_ptr<plan_fftw<scalar_type, dir>>(new plan_fftw<scalar_type, dir>(size, howmany, stride, rdist, cdist));
    }

    mutable int size, howmany, stride, blocks;
    mutable int rdist, cdist, rblock_stride, cblock_stride, rsize, csize;
    mutable std::unique_ptr<plan_fftw<float, direction::forward>> sforward;
    mutable std::unique_ptr<plan_fftw<double, direction::forward>> dforward;
    mutable std::unique_ptr<plan_fftw<float, direction::backward>> sbackward;
    mutable std::unique_ptr<plan_fftw<double, direction::backward>> dbackward;
};

template<> struct one_dim_backend<backend::fftw>{
    using type = fftw_executor;
    using type_r2c = fftw_executor_r2c;

    static std::unique_ptr<fftw_executor> make(box3d const box, int dimension){
        return std::unique_ptr<fftw_executor>(new fftw_executor(box, dimension));
    }

    static std::unique_ptr<fftw_executor_r2c> make_r2c(box3d const box, int dimension){
        return std::unique_ptr<fftw_executor_r2c>(new fftw_executor_r2c(box, dimension));
    }
};

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
