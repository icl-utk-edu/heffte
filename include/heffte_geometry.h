/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_GEOMETRY_H
#define HEFFTE_GEOMETRY_H

#include "heffte_utils.h"

#include <iostream>
#include <ostream>

namespace heffte {

/*!
 * \brief A generic container that describes a 3d box of indexes.
 */
struct box3d{
    //! \brief Constructs a box from the low and high indexes, the span in each direction includes the low and high.
    box3d(std::array<int, 3> clow, std::array<int, 3> chigh) :
        low(clow), high(chigh), size({high[0] - low[0] + 1, high[1] - low[1] + 1, high[2] - low[2] + 1})
    {}
    //! \brief Returns true if the box contains no indexes.
    bool empty() const{ return (size[0] <= 0 or size[1] <= 0 or size[2] <= 0); }
    //! \brief Counts all indexes in the box, i.e., the volume.
    int count() const{ return (empty()) ? 0 : (size[0] * size[1] * size[2]); }
    //! \brief Creates a box that holds the intersection of this box and the \b other.
    box3d collide(box3d const other) const{
        return box3d({std::max(low[0], other.low[0]), std::max(low[1], other.low[1]), std::max(low[2], other.low[2])},
                     {std::min(high[0], other.high[0]), std::min(high[1], other.high[1]), std::min(high[2], other.high[2])});
    }
    //! \brief Compares two boxes, returns \b true if either of the box boundaries do not match.
    bool operator != (box3d const &other) const{
        for(int i=0; i<3; i++)
            if (low[i] != other.low[i] or high[i] != other.high[i]) return true;
        return false;
    }
    std::array<int, 3> const low, high, size;
};

inline std::ostream & operator << (std::ostream &os, box3d const box){
    for(int i=0; i<3; i++)
        os << box.low[i] << "  " << box.high[i] << "  (" << box.size[i] << ")\n";
    os << "\n";
    return os;
}

struct ioboxes{
    std::vector<box3d> in, out;
};

/*!
 * \brief Returns the box that encapsulates all other boxes.
 *
 * Searches through the world.in boxes and computes the highest and lowest of all entries.
 *
 * \param world the collection of all input and output boxes.
 */
inline box3d find_world(ioboxes const &world){
    std::array<int, 3> low  = world.in[0].low;
    std::array<int, 3> high = world.in[0].high;
    for(auto b : world.in){
        for(int i=0; i<3; i++)
            low[i] = std::min(low[i], b.low[i]);
        for(int i=0; i<3; i++)
            high[i] = std::max(low[i], b.low[i]);
    }
    return {low, high};
}

/*!
 * \brief Returns true if the geometry of the world is as expected.
 *
 * Runs simple checks to ensure that the inboxes will fill the world.
 * \param boxes is the collection of all world boxes
 * \param world the box that incorporates all other boxes
 *
 * The check is not very rigorous at the moment, a true rigorous test
 * will probably be too expensive unless lots of thought is put into it.
 */
inline bool world_complete(ioboxes const &boxes, box3d const world){
    long long wsize = 0;
    for(auto b : boxes.in) wsize += b.count();
    if (wsize < world.count())
        throw std::invalid_argument("The provided input boxes do not fill the world box!");

    for(size_t i=0; i<boxes.in.size(); i++)
        for(size_t j=0; j<boxes.in.size(); j++)
            if (!boxes.in[i].collide(boxes.in[j]).empty())
                throw std::invalid_argument("Input boxes cannot overlap!");

    return true;
}

namespace mpi {
/*!
 * \brief Gather all boxes across all ranks in the comm.
 *
 * Constructs an \b ioboxes struct with input and output boxes collected from all ranks.
 * \param my_inbox is the input box on this rank
 * \param my_outbox is the output box on this rank
 * \param comm is the communicator with all ranks
 *
 * \returns an \b ioboxes struct that holds all boxes across all ranks in the comm
 *
 * Uses MPI_Allgather().
 */
inline ioboxes gather_boxes(box3d const my_inbox, box3d const my_outbox, MPI_Comm const comm){
    std::array<box3d, 2> my_data = {my_inbox, my_outbox};
    std::vector<box3d> all_boxes(2 * mpi::comm_size(comm), box3d({0, 0, 0}, {0, 0, 0}));
    MPI_Allgather(&my_data, 2 * sizeof(box3d), MPI_BYTE, all_boxes.data(), 2 * sizeof(box3d), MPI_BYTE, comm);
    ioboxes result;
    for(auto i = all_boxes.begin(); i < all_boxes.end(); i += 2){
        result.in.push_back(*i);
        result.out.push_back(*(i+1));
    }
    return result;
}

}

}


#endif /* HEFFTE_GEOMETRY_H */
