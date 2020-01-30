/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

// 3d scale ffts GPU library
#ifndef HEFFTE_SCALE_H
#define HEFFTE_SCALE_H

template <class T>
void scale_ffts_gpu(int n, T *data, T fnorm);

#endif /* end if defined SCALE_H */
