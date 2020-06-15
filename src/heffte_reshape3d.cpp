/**
 * @class
 * CPU functions of HEFFT
 */
 /*
     -- HEFFTE (version 0.2) --
        Univ. of Tennessee, Knoxville
        @date
 */

#include "heffte_reshape3d.h"

namespace heffte {

#ifdef Heffte_ENABLE_TRACING

    std::deque<event> event_log;
    std::string log_filename;

#endif


/*!
 * \brief Counts how many boxes from the list have a non-empty intersection with the reference box.
 */
int count_collisions(std::vector<box3d> const &boxes, box3d const reference){
    return std::count_if(boxes.begin(), boxes.end(), [&](box3d const b)->bool{ return not reference.collide(b).empty(); });
}

/*!
 * \brief Returns the ranks that will participate in an all-to-all communication.
 *
 * In a reshape algorithm, consider all ranks and connected them into a graph, where each edge
 * corresponds to a piece of data that must be communicated (send or receive).
 * Then take this rank (defined by the list of send and recv procs) and find the larges connected sub-graph.
 * That corresponds to all the processes that need to participate in an all-to-all communication pattern.
 *
 * \param send_proc is the list of ranks that need data from this rank
 * \param recv_proc is the list of ranks that need to send data to this rank
 * \param input_boxes is the list of all boxes held currently across the comm
 * \param output_boxes is the list of all boxes at the end of the communication
 *
 * \returns a list of ranks that must participate in an all-to-all communication
 */
std::vector<int> a2a_group(std::vector<int> const &send_proc, std::vector<int> const &recv_proc,
                           std::vector<box3d> const &input_boxes, std::vector<box3d> const &output_boxes){
    assert(input_boxes.size() == output_boxes.size());
    std::vector<int> result;
    std::vector<bool> marked(input_boxes.size(), false);

    // start with the processes that are connected to this rank
    for(auto p : send_proc){
        if (marked[p]) continue;
        marked[p] = true;
        result.push_back(p);
    }
    for(auto p : recv_proc){
        if (marked[p]) continue;
        marked[p] = true;
        result.push_back(p);
    }

    // loop over procs in result
    // collide each input_boxes extent with all Nprocs output extents
    // collide each output_boxes extent with all Nprocs input extents
    // add any new collision to result
    // keep iterating until nothing is added to result
    bool adding = true;
    while(adding){
        size_t num_current = result.size();
        for(size_t i=0; i<num_current; i++){
            int iproc = result[i];
            // note the O(n^2) graph search, but should be OK for now
            for(size_t j=0; j<input_boxes.size(); j++){
                if (not marked[j] and not input_boxes[iproc].collide(output_boxes[j]).empty()){
                    result.push_back(j);
                    marked[j] = true;
                }
                if (not marked[j] and not output_boxes[iproc].collide(input_boxes[j]).empty()){
                    result.push_back(j);
                    marked[j] = true;
                }
            }
        }
        adding = (num_current != result.size()); // if nothing got added
    }

    // sort based on the flag
    result.resize(0);
    for(size_t i=0; i<input_boxes.size(); i++)
        if (marked[i]) result.push_back(i);

    return result;
}

/*
 * Assumes that all boxes have the same order which may be different from (0, 1, 2).
 * The data-movement will be done from a contiguous buffer into the lines of a box.
 */
void compute_overlap_map_direct_pack(int me, int nprocs, box3d const source, std::vector<box3d> const &boxes,
                                     std::vector<int> &proc, std::vector<int> &offset, std::vector<int> &sizes, std::vector<pack_plan_3d> &plans){
    for(int i=0; i<nprocs; i++){
        int iproc = (i + me + 1) % nprocs;
        box3d overlap = source.collide(boxes[iproc]);
        if (not overlap.empty()){
            proc.push_back(iproc);
            offset.push_back((overlap.low[source.order[2]] - source.low[source.order[2]]) * source.osize(0) * source.osize(1)
                              + (overlap.low[source.order[1]] - source.low[source.order[1]]) * source.osize(0)
                              + (overlap.low[source.order[0]] - source.low[source.order[0]]));

            plans.push_back({{overlap.osize(0), overlap.osize(1), overlap.osize(2)}, // fast, mid, and slow sizes
                             source.osize(0), source.osize(1) * source.osize(0), // line and plane strides
                             0, 0, {0, 0, 0}});  // ignore the transpose parameters
            sizes.push_back(overlap.count());
        }
    }
}

void compute_overlap_map_transpose_pack(int me, int nprocs, box3d const destination, std::vector<box3d> const &boxes,
                                        std::vector<int> &proc, std::vector<int> &offset, std::vector<int> &sizes, std::vector<pack_plan_3d> &plans){
    for(int i=0; i<nprocs; i++){
        int iproc = (i + me + 1) % nprocs;
        box3d overlap = destination.collide(boxes[iproc]);
        if (not overlap.empty()){
            proc.push_back(iproc);
            offset.push_back((overlap.low[destination.order[2]] - destination.low[destination.order[2]]) * destination.osize(0) * destination.osize(1)
                              + (overlap.low[destination.order[1]] - destination.low[destination.order[1]]) * destination.osize(0)
                              + (overlap.low[destination.order[0]] - destination.low[destination.order[0]]));

            // figure out the map between the fast-mid-slow directions of the destination and the fast-mid-slow directions of the source
            std::array<int, 3> map = {-1, -1, -1};
            for(int j=0; j<3; j++)
                for(int i=0; i<3; i++)
                    if (destination.order[i] == boxes[iproc].order[j])
                        map[j] = i;

            plans.push_back({{overlap.osize(0), overlap.osize(1), overlap.osize(2)}, // fast, mid, and slow sizes
                             destination.osize(0), destination.osize(1) * destination.osize(0), // destination line and plane strides
                             overlap.size[boxes[iproc].order[0]], // strides for the buffer from the received data
                             overlap.size[boxes[iproc].order[0]] * overlap.size[boxes[iproc].order[1]],
                             map});  // map of the sizes of the overlap to the fast, mid and slow directions of the input
            sizes.push_back(overlap.count());
        }
    }
}

template<typename backend_tag, template<typename device> class packer>
reshape3d_alltoallv<backend_tag, packer>::reshape3d_alltoallv(
                        int input_size, int output_size,
                        MPI_Comm master_comm, std::vector<int> const &pgroup,
                        std::vector<int> &&csend_offset, std::vector<int> &&csend_size, std::vector<int> const &send_proc,
                        std::vector<int> &&crecv_offset, std::vector<int> &&crecv_size, std::vector<int> const &recv_proc,
                        std::vector<pack_plan_3d> &&cpackplan, std::vector<pack_plan_3d> &&cunpackplan
                                                                ) :
    reshape3d_base(input_size, output_size),
    comm(mpi::new_comm_form_group(pgroup, master_comm)), me(mpi::comm_rank(comm)), nprocs(mpi::comm_size(comm)),
    send_offset(std::move(csend_offset)), send_size(std::move(csend_size)),
    recv_offset(std::move(crecv_offset)), recv_size(std::move(crecv_size)),
    send_total(std::accumulate(send_size.begin(), send_size.end(), 0)),
    recv_total(std::accumulate(recv_size.begin(), recv_size.end(), 0)),
    packplan(std::move(cpackplan)), unpackplan(std::move(cunpackplan)),
    send(pgroup, send_proc, send_size),
    recv(pgroup, recv_proc, recv_size)
{}

template<typename backend_tag, template<typename device> class packer>
template<typename scalar_type>
void reshape3d_alltoallv<backend_tag, packer>::apply_base(scalar_type const source[], scalar_type destination[], scalar_type workspace[]) const{

    using buffer_container = typename backend::buffer_traits<backend_tag>::template container<scalar_type>;

    scalar_type *send_buffer = workspace;
    scalar_type *recv_buffer = workspace + input_size;

    packer<typename backend::buffer_traits<backend_tag>::location> packit;

    int offset = 0;

    { add_trace name("packing");
    for(auto isend : send.map){
        if (isend >= 0){ // something to send
            packit.pack(packplan[isend], source + send_offset[isend], send_buffer + offset);
            offset += send_size[isend];
        }
    }
    }

    #ifdef Heffte_ENABLE_CUDA
    // the device_synchronize() is needed to flush the kernels of the asynchronous packing
    if (std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::gpu>::value)
        cuda::synchronize_default_stream();
    #endif

    { add_trace name("all2allv");
    MPI_Alltoallv(send_buffer, send.counts.data(), send.displacements.data(), mpi::type_from<scalar_type>(),
                  recv_buffer, recv.counts.data(), recv.displacements.data(), mpi::type_from<scalar_type>(),
                  comm);
    }

    offset = 0;
    { add_trace name("unpacking");
    for(auto irecv : recv.map){
        if (irecv >= 0){ // something received
            packit.unpack(unpackplan[irecv], recv_buffer + offset, destination + recv_offset[irecv]);
            offset += recv_size[irecv];
        }
    }
    }
}

template<typename backend_tag, template<typename device> class packer>
std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>
make_reshape3d_alltoallv(std::vector<box3d> const &input_boxes,
                         std::vector<box3d> const &output_boxes,
                         MPI_Comm const comm){

    int const me = mpi::comm_rank(comm);
    int const nprocs = mpi::comm_size(comm);

    std::vector<pack_plan_3d> packplan, unpackplan; // will be moved into the class
    std::vector<int> send_offset;
    std::vector<int> send_size;
    std::vector<int> send_proc;
    std::vector<int> recv_offset;
    std::vector<int> recv_size;
    std::vector<int> recv_proc;

    box3d outbox = output_boxes[me];
    box3d inbox  = input_boxes[me];

    // number of ranks that need data from me
    int nsend = count_collisions(output_boxes, inbox);

    if (nsend > 0) // if others need something from me, prepare the corresponding sizes and plans
        compute_overlap_map_direct_pack(me, nprocs, input_boxes[me], output_boxes, send_proc, send_offset, send_size, packplan);

    // number of ranks that I need data from
    int nrecv = count_collisions(input_boxes, outbox);

    if (nrecv > 0){ // if I need something from others, prepare the corresponding sizes and plans
        // the transpose logic is included in the unpack procedure, direct_packer does not transpose
        if (std::is_same<packer<backend_tag>, direct_packer<backend_tag>>::value){
            compute_overlap_map_direct_pack(me, nprocs, output_boxes[me], input_boxes, recv_proc, recv_offset, recv_size, unpackplan);
        }else{
            compute_overlap_map_transpose_pack(me, nprocs, output_boxes[me], input_boxes, recv_proc, recv_offset, recv_size, unpackplan);
        }
    }

    return std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>(new reshape3d_alltoallv<backend_tag, packer>(
        inbox.count(), outbox.count(),
        comm, a2a_group(send_proc, recv_proc, input_boxes, output_boxes),
        std::move(send_offset), std::move(send_size), send_proc,
        std::move(recv_offset), std::move(recv_size), recv_proc,
        std::move(packplan), std::move(unpackplan)
                                                       ));
}

template<typename backend_tag, template<typename device> class packer>
reshape3d_pointtopoint<backend_tag, packer>::reshape3d_pointtopoint(
                        int input_size, int output_size, MPI_Comm ccomm,
                        std::vector<int> &&csend_offset, std::vector<int> &&csend_size, std::vector<int> &&csend_proc,
                        std::vector<int> &&crecv_offset, std::vector<int> &&crecv_size, std::vector<int> &&crecv_proc,
                        std::vector<int> &&crecv_loc,
                        std::vector<pack_plan_3d> &&cpackplan, std::vector<pack_plan_3d> &&cunpackplan
                                                                ) :
    reshape3d_base(input_size, output_size), comm(ccomm),
    me(mpi::comm_rank(comm)), nprocs(mpi::comm_size(comm)),
    self_to_self(not crecv_proc.empty() and (crecv_proc.back() == me)), // check whether we should include "me" in the communication scheme
    requests(crecv_proc.size() + ((self_to_self) ? -1 : 0)), // remove 1 if using self-to-self
    send_proc(std::move(csend_proc)), send_offset(std::move(csend_offset)), send_size(std::move(csend_size)),
    recv_proc(std::move(crecv_proc)), recv_offset(std::move(crecv_offset)), recv_size(std::move(crecv_size)),
    recv_loc(std::move(crecv_loc)),
    send_total(std::accumulate(send_size.begin(), send_size.end(), 0)),
    recv_total(std::accumulate(recv_size.begin(), recv_size.end(), 0)),
    packplan(std::move(cpackplan)), unpackplan(std::move(cunpackplan))
{}

template<typename backend_tag, template<typename device> class packer>
template<typename scalar_type>
void reshape3d_pointtopoint<backend_tag, packer>::apply_base(scalar_type const source[], scalar_type destination[], scalar_type workspace[]) const{

    using buffer_container = typename backend::buffer_traits<backend_tag>::template container<scalar_type>;

    scalar_type *send_buffer = workspace;
    scalar_type *recv_buffer = workspace + input_size;

    packer<typename backend::buffer_traits<backend_tag>::location> packit;

    // queue the receive messages, using asynchronous receive
    for(size_t i=0; i<requests.size(); i++){
        heffte::add_trace name("irecv " + std::to_string(recv_size[i]) + " from " + std::to_string(recv_proc[i]));
        MPI_Irecv(recv_buffer + recv_loc[i], recv_size[i], mpi::type_from<scalar_type>(), recv_proc[i], 0, comm, &requests[i]);
    }

    // perform the send commands, using blocking send
    for(size_t i=0; i<send_proc.size() + ((self_to_self) ? -1 : 0); i++){
        { heffte::add_trace name("packing");
        packit.pack(packplan[i], &source[send_offset[i]], send_buffer);
        }

        #ifdef Heffte_ENABLE_CUDA
        // the device_synchronize() is needed to flush the kernels of the asynchronous packing
        if (std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::gpu>::value)
            cuda::synchronize_default_stream();
        #endif

        { heffte::add_trace name("send " + std::to_string(send_size[i]) + " for " + std::to_string(send_proc[i]));
        MPI_Send(send_buffer, send_size[i], mpi::type_from<scalar_type>(), send_proc[i], 0, comm);
        }
    }

    if (self_to_self){ // if using self-to-self, do not invoke an MPI command
        { heffte::add_trace name("self packing");
        packit.pack(packplan.back(), source + send_offset.back(), recv_buffer + recv_loc.back());
        }

        { heffte::add_trace name("self unpacking");
        packit.unpack(unpackplan.back(), recv_buffer + recv_loc.back(), destination + recv_offset.back());
        }
    }

    for(size_t i=0; i<requests.size(); i++){
        int irecv;
        { heffte::add_trace name("waitany");
        MPI_Waitany(requests.size(), requests.data(), &irecv, MPI_STATUS_IGNORE);
        }

        { heffte::add_trace name("unpacking from " + std::to_string(irecv));
        packit.unpack(unpackplan[irecv], recv_buffer + recv_loc[irecv], destination + recv_offset[irecv]);
        }
    }

    #ifdef Heffte_ENABLE_CUDA
    if (std::is_same<typename backend::buffer_traits<backend_tag>::location, tag::gpu>::value)
        cuda::synchronize_default_stream();
    #endif
}

template<typename backend_tag, template<typename device> class packer>
std::unique_ptr<reshape3d_pointtopoint<backend_tag, packer>>
make_reshape3d_pointtopoint(std::vector<box3d> const &input_boxes,
                         std::vector<box3d> const &output_boxes,
                         MPI_Comm const comm){

    int const me = mpi::comm_rank(comm);
    int const nprocs = mpi::comm_size(comm);

    std::vector<pack_plan_3d> packplan, unpackplan; // will be moved into the class
    std::vector<int> send_offset;
    std::vector<int> send_size;
    std::vector<int> send_proc;
    std::vector<int> recv_offset;
    std::vector<int> recv_size;
    std::vector<int> recv_proc;
    std::vector<int> recv_loc;

    box3d outbox = output_boxes[me];
    box3d inbox  = input_boxes[me];

    // number of ranks that need data from me
    int nsend = count_collisions(output_boxes, inbox);

    if (nsend > 0) // if others need something from me, prepare the corresponding sizes and plans
        compute_overlap_map_direct_pack(me, nprocs, input_boxes[me], output_boxes, send_proc, send_offset, send_size, packplan);

    // number of ranks that I need data from
    int nrecv = count_collisions(input_boxes, outbox);

    if (nrecv > 0){ // if I need something from others, prepare the corresponding sizes and plans
        // the transpose logic is included in the unpack procedure, direct_packer does not transpose
        if (std::is_same<packer<backend_tag>, direct_packer<backend_tag>>::value){
            compute_overlap_map_direct_pack(me, nprocs, output_boxes[me], input_boxes, recv_proc, recv_offset, recv_size, unpackplan);
        }else{
            compute_overlap_map_transpose_pack(me, nprocs, output_boxes[me], input_boxes, recv_proc, recv_offset, recv_size, unpackplan);
        }
    }

    recv_loc.push_back(0);
    for(int i=0; i<recv_size.size() - 1; i++)
        recv_loc.push_back(recv_loc.back() + recv_size[i]);

    return std::unique_ptr<reshape3d_pointtopoint<backend_tag, packer>>(new reshape3d_pointtopoint<backend_tag, packer>(
        inbox.count(), outbox.count(), comm,
        std::move(send_offset), std::move(send_size), std::move(send_proc),
        std::move(recv_offset), std::move(recv_size), std::move(recv_proc),
        std::move(recv_loc),
        std::move(packplan), std::move(unpackplan)
                                                       ));
}

#define heffte_instantiate_reshape3d_alltoallv(some_backend) \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<float>(float const[], float[], float[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<double>(double const[], double[], double[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<std::complex<float>>(std::complex<float> const[], std::complex<float>[], std::complex<float>[]) const; \
template void reshape3d_alltoallv<some_backend, direct_packer>::apply_base<std::complex<double>>(std::complex<double> const[], std::complex<double> [], std::complex<double> []) const; \
template void reshape3d_alltoallv<some_backend, transpose_packer>::apply_base<float>(float const[], float[], float[]) const; \
template void reshape3d_alltoallv<some_backend, transpose_packer>::apply_base<double>(double const[], double[], double[]) const; \
template void reshape3d_alltoallv<some_backend, transpose_packer>::apply_base<std::complex<float>>(std::complex<float> const[], std::complex<float>[], std::complex<float>[]) const; \
template void reshape3d_alltoallv<some_backend, transpose_packer>::apply_base<std::complex<double>>(std::complex<double> const[], std::complex<double> [], std::complex<double> []) const; \
 \
template std::unique_ptr<reshape3d_alltoallv<some_backend, direct_packer>> \
make_reshape3d_alltoallv<some_backend, direct_packer>(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const); \
template std::unique_ptr<reshape3d_alltoallv<some_backend, transpose_packer>> \
make_reshape3d_alltoallv<some_backend, transpose_packer>(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const); \
 \
template void reshape3d_pointtopoint<some_backend, direct_packer>::apply_base<float>(float const[], float[], float[]) const; \
template void reshape3d_pointtopoint<some_backend, direct_packer>::apply_base<double>(double const[], double[], double[]) const; \
template void reshape3d_pointtopoint<some_backend, direct_packer>::apply_base<std::complex<float>>(std::complex<float> const[], std::complex<float>[], std::complex<float>[]) const; \
template void reshape3d_pointtopoint<some_backend, direct_packer>::apply_base<std::complex<double>>(std::complex<double> const[], std::complex<double> [], std::complex<double> []) const; \
template void reshape3d_pointtopoint<some_backend, transpose_packer>::apply_base<float>(float const[], float[], float[]) const; \
template void reshape3d_pointtopoint<some_backend, transpose_packer>::apply_base<double>(double const[], double[], double[]) const; \
template void reshape3d_pointtopoint<some_backend, transpose_packer>::apply_base<std::complex<float>>(std::complex<float> const[], std::complex<float>[], std::complex<float>[]) const; \
template void reshape3d_pointtopoint<some_backend, transpose_packer>::apply_base<std::complex<double>>(std::complex<double> const[], std::complex<double> [], std::complex<double> []) const; \
 \
template std::unique_ptr<reshape3d_pointtopoint<some_backend, direct_packer>> \
make_reshape3d_pointtopoint<some_backend, direct_packer>(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const); \
template std::unique_ptr<reshape3d_pointtopoint<some_backend, transpose_packer>> \
make_reshape3d_pointtopoint<some_backend, transpose_packer>(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const); \

#ifdef Heffte_ENABLE_FFTW
heffte_instantiate_reshape3d_alltoallv(backend::fftw);
#endif
#ifdef Heffte_ENABLE_MKL
heffte_instantiate_reshape3d_alltoallv(backend::mkl);
#endif
#ifdef Heffte_ENABLE_CUDA
heffte_instantiate_reshape3d_alltoallv(backend::cufft);
#endif

}
