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

struct box3d{
    //! \brief Constructs a box from the low and high indexes, the span in each direction includes the low and high.
    box3d(std::array<int, 3> clow, std::array<int, 3> chigh) :
        low(clow), high(chigh), size({high[0] - low[0] + 1, high[1] - low[1] + 1, high[2] - low[2] + 1})
    {}
    bool empty() const{ return (size[0] <= 0 or size[1] <= 0 or size[2] <= 0); }
    int count() const{ return (empty()) ? 0 : (size[0] * size[1] * size[2]); }
    box3d collide(box3d const other) const{
        return box3d({std::max(low[0], other.low[0]), std::max(low[1], other.low[1]), std::max(low[2], other.low[2])},
                     {std::min(high[0], other.high[0]), std::min(high[1], other.high[1]), std::min(high[2], other.high[2])});
    }
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

namespace mpi {
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
