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
}

/*!
 * \brief Define the location of the pack/unpack data for the fftw backend (cpu).
 */
template<> struct packer_backend<backend::fftw>{
    //! \brief Affirm that the fftw backend is using data on the CPU.
    using mode = tag::cpu;
};

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
        howmany(get_many(box, dimension)),
        stride(get_stride(box, dimension)),
        dist((dimension == 0) ? size : 1),
        blocks((dimension == 1) ? box.size[2] : 1),
        block_stride(box.size[0] * box.size[1]),
        total_size(box.count())
    {}

    static int get_many(box3d const box, int dimension){
        if (dimension == 0) return box.size[1] * box.size[2];
        if (dimension == 1) return box.size[0];
        return box.size[0] * box.size[1];
    }
    static int get_stride(box3d const box, int dimension){
        if (dimension == 0) return 1;
        if (dimension == 1) return box.size[0];
        return box.size[0] * box.size[1];
    }

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

}

#endif

#endif   /* HEFFTE_BACKEND_FFTW_H */
