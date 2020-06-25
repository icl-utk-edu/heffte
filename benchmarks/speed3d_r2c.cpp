/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
       Performance test for 3D FFTs using heFFTe to compute 3D real-to-complex transforms 
*/

// doesn't work right now, logic is too messy at the moment with the in-place r2c
#define BENCH_INPUT std::complex<precision_type>
#undef BENCH_C2C

#include "speed3d.h"
