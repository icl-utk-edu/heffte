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

    template<typename input_type>
    buffer_container<typename fft_output<input_type>::type> forward(buffer_container<input_type> const &input, scale scaling = scale::none){
        if (input.size() < size_inbox())
            throw std::invalid_argument("The input vector is smaller than size_inbox(), i.e., not enough entries provided to fill the inbox.");
        static_assert(std::is_same<input_type, float>::value or std::is_same<input_type, double>::value,
                      "The input to forward() must be real, i.e., either float or double.");
        buffer_container<typename fft_output<input_type>::type> output(size_outbox());
        forward(input.data(), output.data(), scaling);
        return output;
    }

    template<typename input_type, typename output_type>
    void backward(input_type const input[], output_type output[], scale scaling = scale::none) const{
        static_assert((std::is_same<output_type, float>::value and is_ccomplex<input_type>::value)
                   or (std::is_same<output_type, double>::value and is_zcomplex<input_type>::value),
                "Using either an unknown complex type or an incompatible pair of types!");

        standard_transform(convert_to_standart(input), convert_to_standart(output), scaling);
    }

    template<typename scalar_type>
    buffer_container<typename define_standard_type<scalar_type>::type::value_type> backward(buffer_container<scalar_type> const &input, scale scaling = scale::none){
        static_assert(is_ccomplex<scalar_type>::value or is_zcomplex<scalar_type>::value,
                      "Either calling backward() with non-complex input or using an unknown complex type.");
        buffer_container<typename define_standard_type<scalar_type>::type::value_type> result(size_inbox());
        backward(input.data(), result.data(), scaling);
        return result;
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
