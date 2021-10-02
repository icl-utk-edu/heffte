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
    high[box.order[0]] = 4 * box.osize(0) - 1;
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
        for(int i = 0; i < length; i++){
            fft_signal[i] = std::complex<precision>(input[i]);
        }
        fft_signal[length] = 0.0;

        int index = length-1;
        for(int i = length+1; i < 2*length+1; i++){
            fft_signal[i] = std::complex<precision>(-1.0 * input[index]);
            index --;
        }
    }
    template<typename precision>
    static void post_backward(void*, int length, precision const fft_result[], precision result[]){
        for(int i=0; i<length; i++)
            result[i] = fft_result[2*i + 1];
    }
};

struct cpu_buffer_factory{
    template<typename scalar_type>
    static std::vector<scalar_type> make(void*, size_t size){ return std::vector<scalar_type>(size); }
};

template<typename fft_backend_tag, typename prepost_processor, typename buffer_factory>
struct real2real_executor{
    template<typename index>
    real2real_executor(typename backend::device_instance<fft_backend_tag>::stream_type cstream, box3d<index> const box, int dimension) :
        stream(cstream),
        length(box.osize(0)),
        num_batch(box.osize(1) * box.osize(2)),
        total_size(box.count()),
        fft(make_executor_r2c<fft_backend_tag>(stream, make_cos_box(box), dimension))
    {
        assert(dimension == box.order[0]); // supporting only ordered operations (for now)
    }

    template<typename index>
    real2real_executor(typename backend::device_instance<fft_backend_tag>::stream_type cstream, box3d<index> const, int, int) : stream(cstream)
    { throw std::runtime_error("2D cosine transform is not yet implemented!"); }

    template<typename index>
    real2real_executor(typename backend::device_instance<fft_backend_tag>::stream_type cstream, box3d<index> const) : stream(cstream)
    { throw std::runtime_error("3D cosine transform is not yet implemented!"); }

    template<typename scalar_type>
    void forward(scalar_type data[]) const{
        auto temp = buffer_factory::template make<scalar_type>(stream, fft->real_size() + 1);
        auto ctemp = buffer_factory::template make<std::complex<scalar_type>>(stream, fft->complex_size());
        for(int i=0; i<num_batch; i++){
            prepost_processor::pre_forward(stream, length, data + i * length, temp.data() + i * 4 * length);
        }
        fft->forward(temp.data(), ctemp.data());
        for(int i=0; i<num_batch; i++)
            prepost_processor::post_forward(stream, length, ctemp.data() + i * (2 * length + 1), data + i * length);
    }
    template<typename scalar_type>
    void backward(scalar_type data[]) const{
        auto temp = buffer_factory::template make<scalar_type>(stream, fft->real_size());
        auto ctemp = buffer_factory::template make<std::complex<scalar_type>>(stream, fft->complex_size());
        for(int i=0; i<num_batch; i++)
            prepost_processor::pre_backward(stream, length, data + i * length, ctemp.data() + i * (2 * length + 1));
        fft->backward(ctemp.data(), temp.data());
        for(int i=0; i<num_batch; i++)
            prepost_processor::post_backward(stream, length, temp.data() + 4 * i * length, data + i * length);
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

    std::unique_ptr<typename one_dim_backend<fft_backend_tag>::executor_r2c> fft;
};


}

#endif
