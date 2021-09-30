/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFFTE_COS_EXECUTOR_H
#define HEFFFTE_COS_EXECUTOR_H

#include "heffte_pack3d.h"

namespace heffte {

template<typename index>
box3d<index> make_cos_box(box3d<index> const &box){
    std::array<index, 3> high{box.size[0]-1, box.size[1]-1, box.size[2]-1};
    high[box.order[0]] *= 4;
    return box3d<index>(std::array<index, 3>{0, 0, 0}, high, box.order);
}

struct cpu_cos_pre_pos_processor{
    template<typename precision>
    static void pre_forward(void*, int length, precision const input[], precision fft_signal[]){
        for(int i = 0; i < length; i++){
            fft_signal[2*i] = 0;
            fft_signal[2*i+1] = input[i];
        }
        fft_signal[2*length] = 0;
        for(int i = 0; i < 2*length; i++){
            fft_signal[4*length-i] = fft_signal[i];
        }
    }
    template<typename precision>
    static void post_forward(void*, int length, std::complex<precision> const fft_result[], precision result[]){
        for(int i = 0; i < length; i++){
            result[i] = std::real(fft_result[i]);
        }
    }
    template<typename precision>
    static void pre_backward(void*, int length, precision const input[], std::complex<precision> fft_signal[]){
        // TODO
        std::cout << "calling pre_backward " << length << "  " << input << "  " << fft_signal << std::endl;
    }
    template<typename precision>
    static void post_backward(void*, int length, precision const fft_result[], precision result[]){
        // TODO
        std::cout << "calling post_backward " << length << "  " << fft_result << "  " << result << std::endl;
    }
};

template<typename fft_backend_tag, typename cos_processor>
struct cos_executor{
    template<typename index>
    cos_executor(typename backend::device_instance<fft_backend_tag>::stream_type cstream, box3d<index> const box, int dimension) :
        stream(cstream),
        length(box.osize(0)),
        num_batch(box.osize(1) * box.osize(2)),
        total_size(box.count()),
        fft(one_dim_backend<fft_backend_tag>::make_r2c(stream, make_cos_box(box), dimension))
    {
        assert(dimension == box.order[0]); // supporting only ordered operations (for now)
    }

    template<typename index>
    cos_executor(typename backend::device_instance<fft_backend_tag>::stream_type, box3d<index> const, int, int)
    { throw std::runtime_error("2D cosine transform is not yet implemented!"); }

    template<typename index>
    cos_executor(typename backend::device_instance<fft_backend_tag>::stream_type, box3d<index> const)
    { throw std::runtime_error("3D cosine transform is not yet implemented!"); }

    template<typename scalar_type>
    void forward(scalar_type data[]) const{
        auto temp = make_buffer_container<scalar_type>(stream, 4 * total_size);
        for(int i=0; i<num_batch; i++)
            cos_processor::pre_forward(stream, length, data + i * length, temp.data() + 4 * i * length);
        fft->forward(temp.data(), reinterpret_cast<std::complex<scalar_type>*>(temp.data()));
        for(int i=0; i<num_batch; i++)
            cos_processor::post_forward(stream, length, reinterpret_cast<std::complex<scalar_type>*>(temp.data()) + 4 * i * length, data + i * length);
    }
    template<typename scalar_type>
    void backward(scalar_type data[]) const{
        auto temp = make_buffer_container<scalar_type>(stream, 4 * total_size);
        for(int i=0; i<num_batch; i++)
            cos_processor::pre_backward(stream, length, data + i * length, reinterpret_cast<std::complex<scalar_type>*>(temp.data()) + 4 * i * length);
        fft->backward(reinterpret_cast<std::complex<scalar_type>*>(temp.data()), temp.data());
        for(int i=0; i<num_batch; i++)
            cos_processor::post_backward(stream, length, temp.data() + 4 * i * length, data + i * length);
    }

    template<typename precision>
    void forward(precision const[], std::complex<precision>[]) const{
        throw std::runtime_error("Calling cos-transform with real-to-complex data! This should not happen!");
    }
    template<typename precision>
    void backward(std::complex<precision> indata[], precision outdata[]) const{ forward(outdata, indata); }

    //! \brief Returns the size of the box.
    int box_size() const{ return total_size; }

private:
    typename backend::device_instance<fft_backend_tag>::stream_type stream;

    int length, num_batch, total_size;

    std::unique_ptr<typename one_dim_backend<fft_backend_tag>::type_r2c> fft;
};


}

#endif
