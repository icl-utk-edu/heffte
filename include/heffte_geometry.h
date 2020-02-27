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

    for(size_t i=0; i<3; i++)
        if (world.low[i] != 0)
            throw std::invalid_argument("Global box indexing must start from 0!");

    for(size_t i=0; i<boxes.in.size(); i++)
        for(size_t j=0; j<boxes.in.size(); j++)
            if (!boxes.in[i].collide(boxes.in[j]).empty())
                throw std::invalid_argument("Input boxes cannot overlap!");

    return true;
}

/*!
 * \brief Generate all possible factors of a number.
 *
 * Taking an integer \b n, generates a list of all possible way that
 * \b n can be split into a product of two numbers.
 * This is using a brute force method.
 *
 * \param n is the number to factor
 * \returns list of integer factors that multiply to \b n, i.e.,
 * \code
 *  std::vector<std::array<int, 2>> factors = get_factors(n);
 *  for(auto f : factors) assert( f[0] * f[1] == n );
 * \endcode
 *
 * Note: the result is never empty as it always contains (1, n).
 */
inline std::vector<std::array<int, 2>> get_factors(int const n){
    std::vector<std::array<int, 2>> result;
    for(int i=1; i< n/2; i++){
        if (n % i == 0)
            result.push_back({i, n / i});
    }
    return result;
}

/*!
 * \brief Factorize the MPI ranks into a 2-by-2 grid.
 *
 * Considers all possible factorizations of the total number of processors
 * and select the one with the lowest area which heuristically reduces the
 * number of ranks that need to communicate in each of the all-to-all operations.
 *
 * \param num_procs is the total number of processors to factorize
 *
 * \returns the two dimensions of the grid
 */
inline std::array<int, 2> make_procgrid(int const num_procs){
    auto factors = get_factors(num_procs);
    auto area = [](std::array<int, 2> f)->int{ return f[0]*f[1] + f[0] + f[1]; };
    std::array<int, 2> min_array = factors.front();
    int min_area = area(min_array);
    for(auto f : factors){
        int farea = area(f);
        if (farea < min_area){
            min_array = f;
            min_area = farea;
        }
    }
    return min_array;
}

/*!
 * \brief Splits the world box into a set of boxes that will be assigned to a process in the process grid.
 *
 * \param world is a box describing all indexes of consideration,
 *              here there is no assumption on the lower or upper bound of the world
 * \param proc_grid describes a the number of boxes in each dimension
 *
 * \returns a list of non-overlapping boxes with union that fills the world where each box contains
 *          approximately the same number of indexes
 */
inline std::vector<box3d> split_world(box3d const world, std::array<int, 3> const proc_grid){

    auto fast = [=](int i)->int{ return world.low[0] + i * (world.size[0] / proc_grid[0]) + std::min(i, (world.size[0] % proc_grid[0])); };
    auto mid  = [=](int i)->int{ return world.low[1] + i * (world.size[1] / proc_grid[1]) + std::min(i, (world.size[1] % proc_grid[1])); };
    auto slow = [=](int i)->int{ return world.low[2] + i * (world.size[2] / proc_grid[2]) + std::min(i, (world.size[2] % proc_grid[2])); };

    std::vector<box3d> result;
    for(int k = 0; k < proc_grid[2]; k++){
        for(int j = 0; j < proc_grid[1]; j++){
            for(int i = 0; i < proc_grid[0]; i++){
                result.push_back({{fast(i), mid(j), slow(k)}, {fast(i+1)-1, mid(j+1)-1, slow(k+1)-1}});
            }
        }
    }
    return result;
}

/*!
 * \brief Breaks the wold into a grid of pencils and orders the pencils to the ranks that will minimize communication
 *
 * A pencil is a box with one dimension that matches the entire world,
 * a pencils grid is a two dimensional grid of boxes that captures a three dimensional world box.
 *
 * This calls heffte::split_world() and then rearranges the list so that performing a reshape operation
 * from the source to the resulting list will minimize communication.
 * \param world is a box describing all indexes of consideration,
 *              it is assumed that the world is the union of the \b source boxes
 * \param prod_grid gives the number of boxes to use for the two-by-two grid
 * \param dimension is 0, 1, or 2, indicating the direction of orientation of the pencils,
 *                  e.g., dimension 1 means that pencil.size[1] == world.size[1]
 *                  for each pencil in the output list
 * \param source is the current distribution of boxes across MPI ranks,
 *               and will be used as a reference when remapping boxes to ranks
 *
 * \returns a sorted list of boxes that describes
 */
inline std::vector<box3d> make_pencils(box3d const world, std::array<int, 2> const proc_grid, int const dimension, std::vector<box3d> const &source){
    // create a list of boxes ordered in column major format (following the proc_grid box)
    std::vector<box3d> pencils;
    if (dimension == 0){
        pencils = split_world(world, {1, proc_grid[0], proc_grid[1]});
    }else if (dimension == 1){
        pencils = split_world(world, {proc_grid[0], 1, proc_grid[1]});
    }else{ // third dimension
        pencils = split_world(world, {proc_grid[0], proc_grid[1], 1});
    }

    // assign each of the pencils to the position in result that will maximize the overlap
    std::vector<box3d> result;
    result.reserve(pencils.size());
    std::vector<bool> taken(pencils.size(), false);

    for(size_t i=0; i<pencils.size(); i++){
        // for each box in the result, find the box among the pencils
        // that has not been taken and has the largest overlap with the corresponding source box
        int max_overlap = -1;
        size_t max_index = pencils.size();
        for(size_t j=0; j<pencils.size(); j++){
            int overlap = source[i].collide(pencils[j]).count();
            if (not taken[j] and overlap > max_overlap){
                max_overlap = overlap;
                max_index = j;
            }
        }
        assert( max_index < pencils.size() ); // if we found a box
        taken[max_index] = true;
        result.push_back(pencils[max_index]);
    }

    return result;
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
