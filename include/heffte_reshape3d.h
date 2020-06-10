/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

/** @class */
// Reshape3d class

#ifndef HEFFTE_RESHAPE3D_H
#define HEFFTE_RESHAPE3D_H

#include <mpi.h>
#include "heffte_plan_logic.h"
#include "heffte_backend_fftw.h"
#include "heffte_backend_cuda.h"
#include "heffte_backend_mkl.h"


namespace HEFFTE {

  /*!
   * The class Reshape3d is in charge of data reshape, starting from the input data to the first
   * direction, and going to every direction to finalize by reshaping the computed FFT into the output
   * shape. Objects can be created as follows: new Reshape3d(MPI_Comm user_comm)
   * @param user_comm  MPI communicator for the P procs which own the data
   */

template <class U>
class Reshape3d {
 public:
  int collective;         // 0 = point-to-point MPI, 1 = collective all2all
  int packflag;           // 0 = array, 1 = pointer, 2 = memcpy
  int64_t memusage;       // memory usage in bytes

  enum heffte_memory_type_t memory_type;

  Reshape3d(MPI_Comm);
  ~Reshape3d();

  void setup(int in_ilo, int in_ihi, int in_jlo, int in_jhi, int in_klo, int in_khi,
             int out_ilo, int out_ihi, int out_jlo, int out_jhi, int out_klo, int out_khi,
             int nqty, int user_permute, int user_memoryflag, int &user_sendsize, int &user_recvsize);

  template <class T>
  void reshape(T *in, T *out, T *user_sendbuf, T *user_recvbuf);

 private:
  MPI_Comm world;
  int me,nprocs,me_newcomm,nprocs_newcomm;
  int setupflag;

  class Memory *memory;
  class Memory *memory_cpu;
  class Error *error;

  // details of how to perform a 3d reshape

  int permute;                      // permutation setting = 0,1,2
  int memoryflag;                   // 0 = user-provided bufs, 1 = internal

  // point to point communication

  int nrecv;                        // # of recvs from other procs
  int nsend;                        // # of sends to other procs
  int self;                         // whether I send/recv with myself

  int *send_offset;                 // extraction loc for each send
  int *send_size;                   // size of each send message
  int *send_proc;                   // proc to send each message to
  struct pack_plan_3d *packplan;    // pack plan for each send message
  int *recv_offset;                 // insertion loc for each recv
  int *recv_size;                   // size of each recv message
  int *recv_proc;                   // proc to recv each message from
  int *recv_bufloc;                 // offset in scratch buf for each recv
  MPI_Request *request;             // MPI request for each posted recv
  struct pack_plan_3d *unpackplan;  // unpack plan for each recv message

  // collective communication

  int ngroup;                       // # of procs in my collective comm group
  int *pgroup;                      // list of ranks in my comm group
  MPI_Comm newcomm;                 // communicator for my comm group

  int *sendcnts;                    // args for MPI_All2all()
  int *senddispls;
  int *sendmap;
  int *recvcnts;
  int *recvdispls;
  int *recvmap;

  // memory for reshape sends and recvs and All2all
  // either provided by caller or allocated internally

  U *sendbuf;              // send buffer
  U *recvbuf;              // recv buffer

  // which pack & unpack functions to use

  void (*pack)(U *, U *, struct pack_plan_3d *);
  void (*unpack)(U *, U *, struct pack_plan_3d *);

  // collision between 2 regions

  struct extent_3d {
    int ilo,ihi,isize;
    int jlo,jhi,jsize;
    int klo,khi,ksize;
  };

  int collide(struct extent_3d *, struct extent_3d *, struct extent_3d *);
};

}

/*!
 * \ingroup fft3d
 * \addtogroup hefftereshape Reshape operations
 *
 * A reshape operation is one that modifies the distribution of the indexes
 * across an MPI communicator. In a special case, the reshape can correspond
 * to a simple in-node data transpose (i.e., no communication).
 *
 * The reshape operations inherit from a common heffte::reshape3d_base class
 * that defines the apply method for different data-types and the sizes
 * of the input, output, and scratch workspace.
 * Reshape objects are usually wrapped in std::unique_ptr containers,
 * which handles the polymorphic calls at runtime and also indicates
 * the special case of no-reshape when the container is empty.
 */

namespace heffte {

/*!
 * \ingroup hefftereshape
 * \brief Generates an unpack plan where the boxes and the destination do not have the same order.
 *
 * This method does not make any MPI calls, but it uses the set of boxes the define the current distribution of the indexes
 * and computes the overlap and the proc, offset, and sizes vectors for the receive stage of an all-to-all-v communication patterns.
 * In addition, a set of unpack plans is created where the order of the boxes and the destination are different,
 * which will transpose the data. The plan has to be used in conjunction with the transpose packer.
 */
void compute_overlap_map_transpose_pack(int me, int nprocs, box3d const destination, std::vector<box3d> const &boxes,
                                        std::vector<int> &proc, std::vector<int> &offset, std::vector<int> &sizes, std::vector<pack_plan_3d> &plans);

/*!
 * \ingroup hefftereshape
 * \brief Base reshape interface.
 */
class reshape3d_base{
public:
    //! \brief Constructor that sets the input and output sizes.
    reshape3d_base(int cinput_size, int coutput_size) : input_size(cinput_size), output_size(coutput_size){};
    //! \brief Default virtual destructor.
    virtual ~reshape3d_base() = default;
    //! \brief Apply the reshape, single precision.
    virtual void apply(float const source[], float destination[], float workspace[]) const = 0;
    //! \brief Apply the reshape, double precision.
    virtual void apply(double const source[], double destination[], double workspace[]) const = 0;
    //! \brief Apply the reshape, single precision complex.
    virtual void apply(std::complex<float> const source[], std::complex<float> destination[], std::complex<float> workspace[]) const = 0;
    //! \brief Apply the reshape, double precision complex.
    virtual void apply(std::complex<double> const source[], std::complex<double> destination[], std::complex<double> workspace[]) const = 0;

    //! \brief Returns the input size.
    int size_intput() const{ return input_size; }
    //! \brief Returns the output size.
    int size_output() const{ return output_size; }
    //! \brief Returns the workspace size.
    size_t size_workspace() const{ return input_size + output_size; }

protected:
    //! \brief Stores the size of the input.
    int const input_size;
    //! \brief Stores the size of the output.
    int const output_size;
};

/*!
 * \ingroup hefftereshape
 * \brief Returns the maximum workspace size used by the shapers.
 */
inline size_t get_workspace_size(std::array<std::unique_ptr<reshape3d_base>, 4> const &shapers){
    size_t max_size = 0;
    for(auto const &s : shapers) if (s) max_size = std::max(max_size, s->size_workspace());
    return max_size;
}

/*!
 * \ingroup hefftereshape
 * \brief Reshape algorithm based on the MPI_Alltoallv() method.
 *
 * The communication plan for the reshape requires complex initialization,
 * which is put outside of the class into a factory method.
 * An instance of the class can be created only via the factory method
 * heffte::make_reshape3d_alltoallv()
 * which allows for stronger const correctness and reduces memory footprint.
 *
 * \tparam backend_tag is the heffte backend
 * \tparam packer the packer algorithms to use in arranging the sub-boxes into the global send/recv buffer,
 *         will work with either heffte::direct_packer or heffte::transpose_packer
 */
template<typename backend_tag, template<typename device> class packer>
class reshape3d_alltoallv : public reshape3d_base{
public:
    //! \brief Destructor, frees the comm generated by the constructor.
    ~reshape3d_alltoallv(){ mpi::comm_free(comm); }
    //! \brief Factory method, use to construct instances of the class.
    template<typename b, template<typename d> class p> friend std::unique_ptr<reshape3d_alltoallv<b, p>>
    make_reshape3d_alltoallv(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const);

    //! \brief Apply the reshape operations, single precision overload.
    void apply(float const source[], float destination[], float workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision overload.
    void apply(double const source[], double destination[], double workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, single precision complex overload.
    void apply(std::complex<float> const source[], std::complex<float> destination[], std::complex<float> workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision complex overload.
    void apply(std::complex<double> const source[], std::complex<double> destination[], std::complex<double> workspace[]) const override final{
        apply_base(source, destination, workspace);
    }

    //! \brief Templated apply algorithm for all scalar types.
    template<typename scalar_type>
    void apply_base(scalar_type const source[], scalar_type destination[], scalar_type workspace[]) const;

private:
    /*!
     * \brief Private constructor that accepts a set of arrays that have been pre-computed by the factory.
     */
    reshape3d_alltoallv(int input_size, int output_size,
                        MPI_Comm master_comm, std::vector<int> const &pgroup,
                        std::vector<int> &&send_offset, std::vector<int> &&send_size, std::vector<int> const &send_proc,
                        std::vector<int> &&recv_offset, std::vector<int> &&recv_size, std::vector<int> const &recv_proc,
                        std::vector<pack_plan_3d> &&packplan, std::vector<pack_plan_3d> &&unpackplan);

    MPI_Comm const comm;
    int const me, nprocs;

    std::vector<int> const send_offset;   // extraction loc for each send
    std::vector<int> const send_size;     // size of each send message
    std::vector<int> const recv_offset;   // insertion loc for each recv
    std::vector<int> const recv_size;     // size of each recv message
    int const send_total, recv_total;

    std::vector<pack_plan_3d> const packplan, unpackplan;

    struct iotripple{
        std::vector<int> counts, displacements, map;
        iotripple(std::vector<int> const &pgroup, std::vector<int> const &proc, std::vector<int> const &sizes) :
            counts(pgroup.size(), 0), displacements(pgroup.size(), 0), map(pgroup.size(), -1)
        {
            int offset = 0;
            for(size_t src = 0; src < pgroup.size(); src++){
                for(size_t i=0; i<proc.size(); i++){
                    if (proc[i] != pgroup[src]) continue;
                    counts[src] = sizes[i];
                    displacements[src] = offset;
                    offset += sizes[i];
                    map[src] = i;
                }
            }
        }

    };

    iotripple const send, recv;
};

/*!
 * \ingroup hefftereshape
 * \brief Factory method that all the necessary work to establish the communication patterns.
 *
 * The purpose of the factory method is to isolate the initialization code and ensure that the internal
 * state of the class is minimal and const-correct, i.e., objects do not hold onto data that will not be
 * used in a reshape apply and the data is labeled const to prevent accidental corruption.
 *
 * \tparam backend_tag the backend to use for the reshape operations
 * \tparam packer is the packer to use to parts of boxes into global send/recv buffer
 *
 * \param input_boxes list of all input boxes across all ranks in the comm
 * \param output_boxes list of all output boxes across all ranks in the comm
 * \param comm the communicator associated with all the boxes
 *
 * \returns unique_ptr containing an instance of the heffte::reshape3d_alltoallv
 *
 * Note: the input and output boxes associated with this rank are located at position
 * mpi::comm_rank() in the respective lists.
 */
template<typename backend_tag, template<typename device> class packer = direct_packer>
std::unique_ptr<reshape3d_alltoallv<backend_tag, packer>>
make_reshape3d_alltoallv(std::vector<box3d> const &input_boxes,
                         std::vector<box3d> const &output_boxes,
                         MPI_Comm const);

/*!
 * \ingroup hefftereshape
 * \brief Reshape algorithm based on the MPI_Send() and MPI_Irecv() methods.
 *
 * Similar to heffte::reshape3d_alltoallv, this class handles a point-to-point reshape
 * and the initialization can be done only with the heffte::make_reshape3d_pointtopoint() factory.
 *
 * \tparam backend_tag is the heffte backend
 * \tparam packer the packer algorithms to use in arranging the sub-boxes into the global send/recv buffer
 */
template<typename backend_tag, template<typename device> class packer>
class reshape3d_pointtopoint : public reshape3d_base{
public:
    //! \brief Destructor, frees the comm generated by the constructor.
    ~reshape3d_pointtopoint() = default;
    //! \brief Factory method, use to construct instances of the class.
    template<typename b, template<typename d> class p> friend std::unique_ptr<reshape3d_pointtopoint<b, p>>
    make_reshape3d_pointtopoint(std::vector<box3d> const&, std::vector<box3d> const&, MPI_Comm const);

    //! \brief Apply the reshape operations, single precision overload.
    void apply(float const source[], float destination[], float workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision overload.
    void apply(double const source[], double destination[], double workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, single precision complex overload.
    void apply(std::complex<float> const source[], std::complex<float> destination[], std::complex<float> workspace[]) const override final{
        apply_base(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision complex overload.
    void apply(std::complex<double> const source[], std::complex<double> destination[], std::complex<double> workspace[]) const override final{
        apply_base(source, destination, workspace);
    }

    //! \brief Templated apply algorithm for all scalar types.
    template<typename scalar_type>
    void apply_base(scalar_type const source[], scalar_type destination[], scalar_type workspace[]) const;

private:
    /*!
     * \brief Private constructor that accepts a set of arrays that have been pre-computed by the factory.
     */
    reshape3d_pointtopoint(int input_size, int output_size, MPI_Comm ccomm,
                           std::vector<int> &&send_offset, std::vector<int> &&send_size, std::vector<int> &&send_proc,
                           std::vector<int> &&recv_offset, std::vector<int> &&recv_size, std::vector<int> &&recv_proc,
                           std::vector<int> &&recv_loc,
                           std::vector<pack_plan_3d> &&packplan, std::vector<pack_plan_3d> &&unpackplan);

    MPI_Comm const comm;
    int const me, nprocs;
    bool const self_to_self;
    mutable std::vector<MPI_Request> requests; // recv_proc.size() requests, but remove one if using self_to_self communication

    std::vector<int> const send_proc;     // processor to send towards
    std::vector<int> const send_offset;   // extraction loc for each send
    std::vector<int> const send_size;     // size of each send message
    std::vector<int> const recv_proc;     // processor to receive from
    std::vector<int> const recv_offset;   // insertion loc for each recv
    std::vector<int> const recv_size;     // size of each recv message
    std::vector<int> const recv_loc;      // offset in the receive buffer (recv_offset refers to the the destination buffer)
    int const send_total, recv_total;

    std::vector<pack_plan_3d> const packplan, unpackplan;
};

/*!
 * \ingroup hefftereshape
 * \brief Factory method that all the necessary work to establish the communication patterns.
 *
 * The purpose of the factory method is to isolate the initialization code and ensure that the internal
 * state of the class is minimal and const-correct, i.e., objects do not hold onto data that will not be
 * used in a reshape apply and the data is labeled const to prevent accidental corruption.
 *
 * \tparam backend_tag the backend to use for the reshape operations
 * \tparam packer is the packer to use to parts of boxes into global send/recv buffer
 *
 * \param input_boxes list of all input boxes across all ranks in the comm
 * \param output_boxes list of all output boxes across all ranks in the comm
 * \param comm the communicator associated with all the boxes
 *
 * \returns unique_ptr containing an instance of the heffte::reshape3d_pointtopoint
 *
 * Note: the input and output boxes associated with this rank are located at position
 * mpi::comm_rank() in the respective lists.
 */
template<typename backend_tag, template<typename device> class packer = direct_packer>
std::unique_ptr<reshape3d_pointtopoint<backend_tag, packer>>
make_reshape3d_pointtopoint(std::vector<box3d> const &input_boxes,
                            std::vector<box3d> const &output_boxes,
                            MPI_Comm const);

/*!
 * \ingroup hefftereshape
 * \brief Special case of the reshape that does not involve MPI communication but applies a transpose instead.
 *
 * The operations is implemented as a single unpack operation using the transpose_packer with the same location tag.
 */
template<typename location_tag>
class reshape3d_transpose : public reshape3d_base{
public:
    //! \brief Constructor using the provided unpack plan.
    reshape3d_transpose(pack_plan_3d const cplan) :
        reshape3d_base(cplan.size[0] * cplan.size[1] * cplan.size[2], cplan.size[0] * cplan.size[1] * cplan.size[2]),
        plan(cplan)
        {}

    //! \brief Apply the reshape operations, single precision overload.
    void apply(float const source[], float destination[], float workspace[]) const override final{
        transpose(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision overload.
    void apply(double const source[], double destination[], double workspace[]) const override final{
        transpose(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, single precision complex overload.
    void apply(std::complex<float> const source[], std::complex<float> destination[], std::complex<float> workspace[]) const override final{
        transpose(source, destination, workspace);
    }
    //! \brief Apply the reshape operations, double precision complex overload.
    void apply(std::complex<double> const source[], std::complex<double> destination[], std::complex<double> workspace[]) const override final{
        transpose(source, destination, workspace);
    }

private:
    template<typename scalar_type>
    void transpose(scalar_type const *source, scalar_type *destination, scalar_type *workspace) const{
        if (source == destination){ // in-place transpose will need workspace
            data_manipulator<location_tag>::copy_n(source, size_intput(), workspace);
            transpose_packer<location_tag>().unpack(plan, workspace, destination);
        }else{
            transpose_packer<location_tag>().unpack(plan, source, destination);
        }
    }

    pack_plan_3d const plan;
};

/*!
 * \ingroup hefftereshape
 * \brief Factory method to create a reshape3d instance.
 *
 * Creates a reshape operation from the geometry defined by the input boxes to the geometry defined but the output boxes.
 * The boxes are spread across the given MPI communicator where the boxes associated with the current MPI rank is located
 * at input_boxes[mpi::comm_rank(comm)] and output_boxes[mpi::comm_rank(comm)].
 *
 * - If the input and output are the same, then an empty unique_ptr is created.
 * - If the geometries differ only in the order, then a reshape3d_transpose instance is created.
 * - In all other cases, a reshape3d_alltoallv instance is created using either direct_packer or transpose_packer.
 *
 * Assumes that the order of the input and output geometries are consistent, i.e.,
 * input_boxes[i].order == input_boxes[j].order for all i, j.
 */
template<typename backend_tag>
std::unique_ptr<reshape3d_base> make_reshape3d(std::vector<box3d> const &input_boxes,
                                               std::vector<box3d> const &output_boxes,
                                               MPI_Comm const comm,
                                               plan_options const options){
    if (match(input_boxes, output_boxes)){
        if (input_boxes[0].ordered_same_as(output_boxes[0])){
            return std::unique_ptr<reshape3d_base>();
        }else{
            int const me = mpi::comm_rank(comm);
            std::vector<int> proc, offset, sizes;
            std::vector<pack_plan_3d> plans;

            compute_overlap_map_transpose_pack(0, 1, output_boxes[me], {input_boxes[me]}, proc, offset, sizes, plans);

            return std::unique_ptr<reshape3d_base>(new reshape3d_transpose<typename backend::buffer_traits<backend_tag>::location>(plans[0]));
        }
    }else{
        if (options.use_alltoall){
            if (input_boxes[0].ordered_same_as(output_boxes[0])){
                return make_reshape3d_alltoallv<backend_tag, direct_packer>(input_boxes, output_boxes, comm);
            }else{
                return make_reshape3d_alltoallv<backend_tag, transpose_packer>(input_boxes, output_boxes, comm);
            }
        }else{
            if (input_boxes[0].ordered_same_as(output_boxes[0])){
                return make_reshape3d_pointtopoint<backend_tag, direct_packer>(input_boxes, output_boxes, comm);
            }else{
                return make_reshape3d_pointtopoint<backend_tag, transpose_packer>(input_boxes, output_boxes, comm);
            }
        }
    }
}

}

#endif
