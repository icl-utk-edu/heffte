/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_CONFIG_H
#define HEFFTE_CONFIG_H

#define Heffte_VERSION_MAJOR @Heffte_VERSION_MAJOR@
#define Heffte_VERSION_MINOR @Heffte_VERSION_MINOR@
#define Heffte_VERSION_PATCH @Heffte_VERSION_PATCH@

#cmakedefine Heffte_ENABLE_FFTW
#cmakedefine Heffte_ENABLE_MKL
#cmakedefine Heffte_ENABLE_CUDA

#cmakedefine Heffte_ENABLE_TRACING


#endif  /* HEFFTE_CONFIG_H */
