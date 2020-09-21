/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_MAGMA_HELPERS_H
#define HEFFTE_MAGMA_HELPERS_H

#include "heffte_pack3d.h"

namespace heffte{

namespace gpu {
    template<typename> struct magma_handle{
        template<typename scalar_type> void scal(int, double, scalar_type*) const{}
    };
    template<>
    struct magma_handle<tag::gpu>{
        magma_handle();
        ~magma_handle();
        void scal(int num_entries, double scale_factor, float *data) const;
        void scal(int num_entries, double scale_factor, double *data) const;
        template<typename precision_type>
        void scal(int num_entries, double scale_factor, std::complex<precision_type> *data) const{
            scal(2*num_entries, scale_factor, reinterpret_cast<precision_type*>(data));
        }
        mutable void *handle;
    };
}

}

#endif   /* HEFFTE_MAGMA_HELPERS_H */
