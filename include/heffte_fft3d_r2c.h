/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef FFT_FFT3D_R2C_H
#define FFT_FFT3D_R2C_H

#include "heffte_fft3d.h"

namespace heffte {

template<typename backend_tag>
class fft3d_r2c{
public:
    using backend_executor_c2c = typename one_dim_backend<backend_tag>::type;
    using backend_executor_r2c = typename one_dim_backend<backend_tag>::type_r2c;

    template<typename T> using buffer_container = typename backend::buffer_traits<backend_tag>::template container<T>;

    fft3d_r2c(box3d const cinbox, box3d const coutbox, int r2c_direction, MPI_Comm const);

    //! \brief Returns the size of the inbox defined in the constructor.
    int size_inbox() const{ return inbox.count(); }
    //! \brief Returns the size of the outbox defined in the constructor.
    int size_outbox() const{ return outbox.count(); }

    template<typename input_type, typename output_type>
    void forward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<input_type, float>::value and is_ccomplex<output_type>::value)
                   or (std::is_same<input_type, double>::value and is_zcomplex<output_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), scaling);
    }

    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), scaling);
    }

private:
    template<typename scalar_type>
    void standard_transform(scalar_type const input[], std::complex<scalar_type> output[], scale) const;
    template<typename scalar_type>
    void standard_transform(std::complex<scalar_type> const input[], scalar_type output[], scale) const;

    box3d inbox, outbox;
    double scale_factor;
    std::array<std::unique_ptr<reshape3d_base>, 4> forward_shaper;
    std::array<std::unique_ptr<reshape3d_base>, 4> backward_shaper;

    std::unique_ptr<backend_executor_r2c> fft_r2c;
    std::unique_ptr<backend_executor_c2c> fft1, fft2;
};

}

#endif
