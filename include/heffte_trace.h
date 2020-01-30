/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_TRACE_H
#define HEFFTE_TRACE_H

#include "heffte_common.h"
#define heffteMaxGPUs                        1

#define heffte_queue_t                       int
#define heffte_event_t                       int
#define heffte_setdevice(i_)                 ((void)(0))
#define heffte_device_sync()                 ((void)(0))
#define heffte_wtime                         MPI_Wtime
#define heffte_event_create(i_)              ((void)(0))
#define heffte_event_record(i_,j_)           ((void)(0))
#define heffte_event_elapsedtime(i_,j_,k_)   ((void)(0))


#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------
#ifdef TRACING_MPI
void trace_init     ( int nthreads, int ngpus, int nstream, heffte_queue_t *streams );
void trace_cpu_start( int core, const char* tag, const char* label );
void trace_op_count  ( int core, double ops );
void trace_cpu_end  ( int core );
void trace_gpu_start( int core, int stream_num, const char* tag, const char* label );
void trace_gpu_end  ( int core, int stream_num );
void trace_finalize ( const char* filename, const char* cssfile, double nops );
void trace_cpu_increase_cnt( int core );
void trace_cpu_decrease_cnt( int core );
#else
#ifdef HEFFTE_NVTX_TRACING
#include "nvtx_trace.h"
#define trace_init(       x1, x2, x3, x4 ) ((void)(0))
#define trace_cpu_start(  x1, x2, x3     ) nvtx_trace_start(x2, x1)
#define trace_op_count(   x1, x2         ) ((void)(0))
#define trace_cpu_end(    x1             ) nvtx_trace_end()
#define trace_gpu_start(  x1, x2, x3, x4 ) ((void)(0))
#define trace_gpu_end(    x1, x2         ) ((void)(0))
#define trace_finalize(   x1, x2, x3     ) ((void)(0))
#define trace_cpu_increase_cnt( x1 ) ((void)(0))
#define trace_cpu_decrease_cnt( x1 ) ((void)(0))
#else
#define trace_init(       x1, x2, x3, x4 ) ((void)(0))
#define trace_cpu_start(  x1, x2, x3     ) ((void)(0))
#define trace_op_count(   x1, x2         ) ((void)(0))
#define trace_cpu_end(    x1             ) ((void)(0))
#define trace_gpu_start(  x1, x2, x3, x4 ) ((void)(0))
#define trace_gpu_end(    x1, x2         ) ((void)(0))
#define trace_finalize(   x1, x2, x3     ) ((void)(0))

#define trace_cpu_increase_cnt( x1 ) ((void)(0))
#define trace_cpu_decrease_cnt( x1 ) ((void)(0))
#endif
#endif

#ifdef __cplusplus
}
#endif


#endif        //  #ifndef HEFFTE_TRACE_H
