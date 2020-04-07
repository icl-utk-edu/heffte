/**
 * @file
 * Tool for profiling HEFFTE
 */
 /*
     -- HEFFTE (version 0.2) --
        Univ. of Tennessee, Knoxville
        @date
 */

#ifdef TRACING_MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>      // strerror_r
#include <math.h>
#include <set>
#include <string>

#include "heffte_trace.h"
#include "heffte_common.h"

// define TRACING_MPI to compile these functions, e.g.,
// gcc -DTRACING -c trace.cpp
#ifdef TRACING_MPI

// set TRACE_METHOD = 2 to record start time as
// later of CPU time and previous event's end time.
// set TRACE_METHOD = 1 to record start time using CUDA event.
#define TRACE_METHOD 2



// ----------------------------------------
const int MAX_CORES       = 192;                 // CPU cores
const int MAX_GPU_STREAMS = heffteMaxGPUs * 4;  // #devices * #streams per device
const int MAX_EVENTS      = 20000;
const int MAX_LABEL_LEN   = 16;


/*
 * Copy src to string dst of size siz.  At most siz-1 characters
 * will be copied.  Always NUL terminates (unless siz == 0).
 * Returns strlen(src); if retval >= siz, truncation occurred.
 */
extern "C"
size_t
heffte_strlcpy(char *dst, const char *src, size_t siz)
{
    char *d = dst;
    const char *s = src;
    size_t n = siz;

    /* Copy as many bytes as will fit */
    if (n != 0) {
        while (--n != 0) {
            if ((*d++ = *s++) == '\0')
                break;
        }
    }

    /* Not enough room in dst, add NUL and traverse rest of src */
    if (n == 0) {
        if (siz != 0)
            *d = '\0';     /* NUL-terminate dst */
        while (*s++)
            ;
    }

    return (s - src - 1);   /* count does not include NUL */
}
// ----------------------------------------
struct event_log
{
    int    ncore;
    int    cpu_id   [ MAX_CORES ];
    double cpu_first;
    double cpu_start[ MAX_CORES ][ MAX_EVENTS ];
    double cpu_end  [ MAX_CORES ][ MAX_EVENTS ];
    double cpu_op_count  [ MAX_CORES ][ MAX_EVENTS ];
    char   cpu_tag  [ MAX_CORES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
    char   cpu_label[ MAX_CORES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];

    int          ngpu;
    int          nstream;
    heffte_queue_t streams  [ MAX_GPU_STREAMS ];
    int          gpu_id   [ MAX_GPU_STREAMS ];
    heffte_event_t  gpu_first[ MAX_GPU_STREAMS ];
#if TRACE_METHOD == 2
    double       gpu_start[ MAX_GPU_STREAMS ][ MAX_EVENTS ];
#else
    heffte_event_t  gpu_start[ MAX_GPU_STREAMS ][ MAX_EVENTS ];
#endif
    heffte_event_t  gpu_end  [ MAX_GPU_STREAMS ][ MAX_EVENTS ];
    char         gpu_tag  [ MAX_GPU_STREAMS ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
    char         gpu_label[ MAX_GPU_STREAMS ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
};

// global log object
struct event_log glog;


// ----------------------------------------
extern "C"
void trace_init( int ncore, int ngpu, int nstream, heffte_queue_t* streams)
{
    if ( ncore > MAX_CORES ) {
        fprintf( stderr, "Error in trace_init: ncore %d > MAX_CORES %d\n",
                 ncore, MAX_CORES );
        return;
    }
    if ( ngpu*nstream > MAX_GPU_STREAMS ) {
        fprintf( stderr, "Error in trace_init: (ngpu=%d)*(nstream=%d) > MAX_GPU_STREAMS=%d\n",
                 ngpu, nstream, MAX_GPU_STREAMS );
        return;
    }

#ifdef TRACING_MPI
    int flag;
    MPI_Initialized(&flag);
    if ( !flag ) MPI_Init(0, NULL);
#endif

    glog.ncore   = ncore;
    glog.ngpu    = ngpu;
    glog.nstream = nstream;

    memset( glog.cpu_op_count, 0, MAX_CORES*MAX_EVENTS*sizeof(double) );

    // initialize ID = 0
    for( int core = 0; core < ncore; ++core ) {
        glog.cpu_id[core] = 0;
    }
    for( int dev = 0; dev < ngpu; ++dev ) {
        for( int s = 0; s < nstream; ++s ) {
            int t = dev*glog.nstream + s;
            glog.gpu_id[t] = 0;
            glog.streams[t] = streams[t];
        }
        heffte_setdevice( dev );
        heffte_device_sync();

    }
    // now that all GPUs are sync'd, record start time
    for( int dev = 0; dev < ngpu; ++dev ) {
        heffte_setdevice( dev );
        for( int s = 0; s < nstream; ++s ) {
            int t = dev*glog.nstream + s;
            heffte_event_create( &glog.gpu_first[t] );
            heffte_event_record(  glog.gpu_first[t], glog.streams[t] );
        }
    }
    // sync again
    for( int dev = 0; dev < ngpu; ++dev ) {
        heffte_setdevice( dev );
        heffte_device_sync();
    }
    glog.cpu_first = heffte_wtime();

#ifdef TRACING_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    glog.cpu_first = heffte_wtime();
#endif

}


// ----------------------------------------
extern "C"
/**
 * Start profiling CPU routines
 * @param core Current processor
 * @param tag Tag for routine
 * @param lbl Label for routine being profiled
 */
void trace_cpu_start( int core, const char* tag, const char* lbl )
{
    int id = glog.cpu_id[core];
    glog.cpu_start[core][id] = heffte_wtime();
    heffte_strlcpy( glog.cpu_tag  [core][id], tag, MAX_LABEL_LEN );
    heffte_strlcpy( glog.cpu_label[core][id], lbl, MAX_LABEL_LEN );
}

// ----------------------------------------
extern "C"
void trace_op_count( int core, double ops )
{
    int id = glog.cpu_id[core];
    glog.cpu_op_count[core][id] = ops;
}

// ----------------------------------------
extern "C"
/**
 * End profiling CPU routines
 * @param core Current processor
 * @param tag Tag for routine
 * @param lbl Label for routine being profiled
 */
void trace_cpu_end( int core )
{
    int id = glog.cpu_id[core];
    glog.cpu_end[core][id] = heffte_wtime();
    if ( id+1 < MAX_EVENTS ) {
        glog.cpu_id[core] = id+1;
    }
}


// ----------------------------------------
extern "C"
/**
 * Start profiling GPU routines
 * @param core Current processor
 * @param tag Tag for routine
 * @param lbl Label for routine being profiled
 */
void trace_gpu_start( int dev, int s, const char* tag, const char* lbl )
{
    int t = dev*glog.nstream + s;
    int id = glog.gpu_id[t];
#if TRACE_METHOD == 2
    glog.gpu_start[t][id] = heffte_wtime();
#else
    heffte_event_create( &glog.gpu_start[t][id] );
    heffte_event_record(  glog.gpu_start[t][id], glog.streams[t] );
#endif
    heffte_strlcpy( glog.gpu_tag  [t][id], tag, MAX_LABEL_LEN );
    heffte_strlcpy( glog.gpu_label[t][id], lbl, MAX_LABEL_LEN );
}


// ----------------------------------------
extern "C"
/**
 * End profiling GPU routines
 * @param core Current processor
 * @param tag Tag for routine
 * @param lbl Label for routine being profiled
 */
void trace_gpu_end( int dev, int s )
{
    int t = dev*glog.nstream + s;
    int id = glog.gpu_id[t];
    heffte_event_create( &glog.gpu_end[t][id] );
    heffte_event_record(  glog.gpu_end[t][id], glog.streams[t] );
    if ( id+1 < MAX_EVENTS ) {
        glog.gpu_id[t] = id+1;
    }
}


// ----------------------------------------
extern "C"
void trace_finalize( const char* filename, const char* cssfile, double nops )
{
    // these are all in SVG "pixels"
    double xscale = 1000.; // pixels per second
    //double xscale = 100000.; // pixels per second
    double height =  200.;  // of each row
    double margin =  5.;  // page margin and between some elements
    double space  =  5.;  // between rows
    double pad    =  5.;  // around text
    double label  = 500;  // width of "CPU:", "GPU:" labels
    double left   = 2*margin + label;
    double xtick  = 0.5;  // interval of xticks (in seconds)
    char buf[ 1024 ];

    // sync devices
    for( int dev = 0; dev < glog.ngpu; ++dev ) {
        heffte_setdevice( dev );
        heffte_device_sync();
    }
    double mytime  = heffte_wtime() - glog.cpu_first;
    double max_time = mytime;

    int mpisize = 1;
    int mpirank = 0;

    int my_tasks  = 0;
    int max_tasks;

    // find the maximum number of tasks per core (use to scale the figure)
    for( int core = 0; core < glog.ncore; ++core )
        my_tasks = max(my_tasks, glog.cpu_id[core]);
    if ( my_tasks > MAX_EVENTS ) my_tasks = MAX_EVENTS;
    max_tasks = my_tasks;


#ifdef TRACING_MPI
    int ierr;
    MPI_Status mpistatus;
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce( &mytime, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce( &max_tasks, &my_tasks, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if ( mpirank == 0 ) fprintf( stderr, "tracing found MPI grid of size %d\n", mpisize);
#endif

    //----------------------------------------------
    // try to automatically scale the figure
    double avgt_pertask = max_time/(double)max_tasks;

    xscale = 500./avgt_pertask;

    if ( mpirank == 0 ) fprintf( stderr, "xscale %lf max_task %d time %lf\n",xscale,max_tasks,max_time);
    xtick = max_time/4.0;
    //----------------------------------------------


    if ( mpirank == 0 ) {
        FILE* trace_file = fopen( filename, "w" );
        if ( trace_file == NULL ) {
            strerror_r( errno, buf, sizeof(buf) );
            fprintf( stderr, "Can't open file '%s': %s (%d)\n", filename, buf, errno );
            return;
        }
        fprintf( stderr, "writing trace to '%s'\n", filename );

        // row for each CPU and GPU/stream (with space between), time scale, legend
        // 4 margins: at top, above time scale, above legend, at bottom
        int h = (int)( (glog.ncore + glog.ngpu*glog.nstream)*(height + space) - space + 2*height + 4*margin ) * mpisize;
        int w = (int)( left + max_time*xscale + margin );
        fprintf( trace_file, "<!-- azzam_START_TRACE_HERE -->\n");
        fprintf( trace_file, "<!-- azzam_info nnodes= %d  nlevels= %d margin= %d space= %d height= %d  -->\n", mpisize, glog.ncore, (int)margin, (int)space, (int)height );
        fprintf( trace_file, "<!-- azzam_START_HEADER_TRACE_HERE -->\n");
        fprintf( trace_file,
             "<?xml version=\"1.0\" standalone=\"no\"?>\n"
                 "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
                 "    \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n\n"
                 "<svg version=\"1.1\" baseProfile=\"full\"\n"
                 "    xmlns=\"http://www.w3.org/2000/svg\"\n"
                 "    xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\"\n"
                 "    viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" preserveAspectRatio=\"none\">\n\n",
                 w, h, w, h );

        // Inkscape does not currently (Jan 2012) support external CSS;
        // see http://wiki.inkscape.org/wiki/index.php/CSS_Support
        // So embed CSS file here
        FILE* css_file = fopen( cssfile, "r" );
        if ( css_file == NULL ) {
            strerror_r( errno, buf, sizeof(buf) );
            fprintf( stderr, "Can't open file '%s': %s; skipping CSS\n", cssfile, buf );
        }
        else {
            fprintf( trace_file, "<style type=\"text/css\">\n" );
            while( fgets( buf, sizeof(buf), css_file ) != NULL ) {
                fputs( buf, trace_file );
            }
            fclose( css_file );
            fprintf( trace_file, "</style>\n\n" );
        }
        fprintf( trace_file, "<!-- azzam_END_HEADER_TRACE_HERE -->\n");
        fprintf( trace_file, "\n\n<!-- azzam_START_LOGEVENT_TRACE_HERE -->\n\n");

        // format takes: x, y, width, height, class (tag), id (label)
        const char* format =
            "<rect x=\"%8.3f\" y=\"%4.0f\" width=\"%8.3f\" height=\"%2.0f\" class=\"%-8s\" inkscape:label=\"TIME:%6.1fms %s\"/>\n";

        // accumulate unique legend entries
        std::set< std::string > legend;

        double top = margin;

        // gather and output events from each node
        for ( int mpinode=0; mpinode<mpisize; mpinode++ ) {

            // receive non-local information
#ifdef TRACING_MPI
            if ( mpinode!=0 ) {
                ierr = MPI_Recv( &glog, sizeof(struct event_log), MPI_BYTE, mpinode, 11, MPI_COMM_WORLD, &mpistatus );
                if(ierr != MPI_SUCCESS){
                    fprintf(stderr, "WARNING MPI RECV from node %d FAILED\n",mpinode);
                }
               // top += (height + margin);
            }
#endif

            // output CPU events
            for( int core = 0; core < glog.ncore; ++core ) {
                if ( glog.cpu_id[core] > MAX_EVENTS ) {
                    glog.cpu_id[core] += 1;  // count last event
                    fprintf( stderr, "WARNING: trace on core %d, reached limit of %d events; output will be truncated.\n",
                             core, glog.cpu_id[core] );
                }
                fprintf( trace_file, "<!-- azzam_node%d_level%d_start_event node %d level %d, nevents %d -->\n", mpinode, core, mpinode, core, glog.cpu_id[core] );
                fprintf( trace_file, "<!-- azzam_node%d_level%d_row -->",mpinode,core);
                fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"node %d level %d\">\n", mpinode, core );
                fprintf( trace_file, "<!-- azzam_node%d_level%d_box1 -->",mpinode,core);
                fprintf( trace_file, "<text x=\"%8.3f\" y=\"%4.0f\" width=\"%4.0f\" height=\"%2.0f\" class=\"mytextfont1\">NODE %d LVL %d:</text>\n",
                         margin,
                         top,
                         label, height,
                         mpinode, core );
                for( int i = 0; i < glog.cpu_id[core]; ++i ) {
                    double start  = glog.cpu_start[core][i] - glog.cpu_first;
                    double end    = glog.cpu_end  [core][i] - glog.cpu_first;
                    fprintf( trace_file, "<!-- azzam_node%d_level%d_box1 -->",mpinode,core);
                    fprintf( trace_file, format,
                             left+start*xscale,
                             top,
                             (end - start)*xscale,
                             height,
                             glog.cpu_tag[core][i],
                             (end-start)*1000,
                             //(end-start==0 ? 0 : (glog.cpu_op_count[core][i]*1e-9)/(end-start)),
                             glog.cpu_label[core][i] );
                    legend.insert( glog.cpu_tag[core][i] );
                }
                top += (height + space);
                fprintf( trace_file, "<!-- azzam_node%d_level%d_row -->",mpinode,core);
                fprintf( trace_file, "</g>\n" );
                fprintf( trace_file, "<!-- azzam_node%d_level%d_end_event node %d level %d, nevents %d -->\n\n", mpinode, core, mpinode, core, glog.cpu_id[core] );
            }
        }
        fprintf( trace_file, "\n\n<!-- azzam_END_LOGEVENT_TRACE_HERE -->\n\n");

        fprintf( trace_file, "\n\n<!-- azzam_START_LEGEND_TRACE_HERE -->\n\n");
        // output time scale
        top += (-space + margin);
        fprintf( trace_file, "<!-- azzam_timelegend -->");
        fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"scale\">\n" );
        fprintf( trace_file, "<!-- azzam_timelegend -->");
        fprintf( trace_file, "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\" class=\"mytextfont2\" >Time (sec):</text>\n",
                 margin, top, label, height );
        for( double s=0; s < max_time; s += xtick ) {
            fprintf( trace_file, "<!-- azzam_timelegend_box2 -->");
            fprintf( trace_file, "<line x1=\"%8.1f\" y1=\"0\" x2=\"%8.1f\" y2=\"%4.0f\" class=\"mytextfont2\"/>\n",
                     left + s*xscale,
                     left + s*xscale,
                     top);
            fprintf( trace_file, "<!-- azzam_timelegend_box1 -->");
            fprintf( trace_file, "<text x=\"%8.1f\" y=\"%4.0f\" >%4.1f</text>\n",
                     left + s*xscale,
                     top,
                     s );
        }
        fprintf( trace_file, "<!-- azzam_timelegend -->");
        fprintf( trace_file, "</g>\n\n" );
        top += (height + margin);

        // output legend
        fprintf( trace_file, "<!-- azzam_legend -->");
        fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"legend\">\n" );
        fprintf( trace_file, "<!-- azzam_legend -->");
        fprintf( trace_file, "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\"class=\"mytextfont3\">Legend:</text>\n",
                 margin, top, label, height );
        double x=left;
        for( std::set<std::string>::iterator it=legend.begin(); it != legend.end(); ++it ) {
            fprintf( trace_file, "<!-- azzam_legend_box1 -->");
            fprintf( trace_file,
                    "<rect x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\" class=\"%s\"/>\n",
                     x,       top, label, height, (*it).c_str());

            fprintf( trace_file, "<!-- azzam_legend_box1 -->");
            fprintf( trace_file,
                     "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\" class=\"mytextfont3\" >%s</text>\n",
                     x + pad, top, label, height, (*it).c_str() );

            x += label + margin;
        }
        fprintf( trace_file, "<!-- azzam_legend -->");
        fprintf( trace_file, "</g>\n\n" );

        fprintf( trace_file, "\n\n<!-- azzam_END_LEGEND_TRACE_HERE -->\n\n");

        if(nops >0){
            double flop_rate = nops*1e-9 / mytime ;
            fprintf(trace_file, "time : %10.4lf, flops : %10.4lf Gflops\n", mytime, flop_rate);
        }
        fprintf( trace_file, "<!-- azzam_end_trace -->");
        fprintf( trace_file, "</svg>\n" );
        fprintf( trace_file, "<!-- azzam_END_TRACE_HERE -->\n");

        fclose( trace_file );

    } else if ( mpirank != 0 ) {

#ifdef TRACING_MPI
        ierr = MPI_Send( &glog, sizeof(struct event_log), MPI_BYTE, 0, 11, MPI_COMM_WORLD );
        if(ierr != MPI_SUCCESS){
            fprintf(stderr, "WARNING MPI SEND from node %d FAILED\n",mpirank);
        }
#endif

    }

}

#endif // TRACING_MPI
