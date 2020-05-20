/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_plan_logic.h"

namespace heffte {

/*!
 * \brief Returns either 0, 1, 2, so that it does not match any of the current values.
 */
inline int get_any_valid(std::array<int, 3> current){
    for(int i=0; i<3; i++){
        bool is_valid = true;
        for(int j=0; j<3; j++)
            if (i == current[j]) is_valid = false;
        if (is_valid) return i;
    }
    return -1; // must never happen
}

/*!
 * \brief Checks if using pencils in multiple directions simultaneously.
 */
inline bool is_pencils(box3d const world, std::vector<box3d> const &shape, std::vector<int> const directions){
    for(auto d : directions) if (not is_pencils(world, shape, d)) return false;
    return true;
}

/*!
 * \brief Applies the r2c direction reduction to the set of boxes.
 */
inline std::vector<box3d> apply_r2c(std::vector<box3d> const &shape, int r2c_direction){
    if (r2c_direction == -1) return shape;
    std::vector<box3d> result;
    for(auto const &s : shape) result.push_back(s.r2c(r2c_direction));
    return result;
}

/*!
 * \brief Checks whether all boxes in the shape have the same order.
 */
inline bool order_is_identical(std::vector<box3d> const &shape){
    if (shape.empty()) return true;
    std::array<int, 3> order = shape.front().order;
    for(auto const &s : shape)
        for(int i=0; i<3; i++)
            if (s.order[i] != order[i]) return false;
    return true;
}


/*!
 * \brief Creates the next box geometry that corresponds to pencils in the given dimension.
 *
 * Similar to heffte::make_pencils(), splits the \b world into a two dimensional processor grid
 * defined by \b proc_grid so that the boxes form pencils in the given third \b dimension.
 * The \b source corresponds to the current processor grid and is needed to that the next
 * set of boxes will be arranges in way that will improve the overlap with the old set.
 * If \b use_reorder is set, the new configuration will be transposed so that the \b dimension
 * will be the new leading (fast) direction.
 *
 * The \b world_out and \b boxes_out correspond to the output geometry of the FFT transformation.
 * If the final geometry satisfies the pencil and order requirements, and if the geometry also
 * gives pencils in the \b test_directions, then it is best to directly jump to the final
 * configuration.
 */
inline std::vector<box3d> next_pencils_shape(box3d const world,
                                             std::array<int, 2> const proc_grid,
                                             int const dimension,
                                             std::vector<box3d> const &source,
                                             bool const use_reorder,
                                             box3d const world_out,
                                             std::vector<int> const test_directions,
                                             std::vector<box3d> const &boxes_out){
    if (use_reorder){
        if (is_pencils(world_out, boxes_out, test_directions)){
            // the boxed_out form the required pencil geometry, but the order may be different
            if (boxes_out[0].order[0] == dimension){ // even the order is good
                return boxes_out;
            }else{
                return reorder(boxes_out, {dimension, (dimension + 1) % 3, (dimension + 2) % 3});
            }
        }else{
            return make_pencils(world, proc_grid, dimension, source,
                                {dimension, (dimension + 1) % 3, (dimension + 2) % 3});
        }
    }else{
        return (is_pencils(world_out, boxes_out, test_directions) ?
                    boxes_out :
                    make_pencils(world, proc_grid, dimension, source, world.order));
    }
}


logic_plan3d plan_operations(ioboxes const &boxes, int r2c_direction, plan_options const opts){

    // form the two world boxes
    box3d const world_in  = find_world(boxes.in);
    box3d const world_out = find_world(boxes.out);

    // perform basic sanity check
    assert( world_complete(boxes.in,  world_in) );
    assert( world_complete(boxes.out, world_out) );
    if (r2c_direction == -1){
        assert( world_in == world_out );
    }else{
        assert( world_in.r2c(r2c_direction) == world_out );
    }
    assert( order_is_identical(boxes.in) );
    assert( order_is_identical(boxes.out) );

    // the 2-d grid of pencils
    std::array<int, 2> proc_grid = make_procgrid(static_cast<int>(boxes.in.size()));

    std::array<int, 3> fft_direction = {-1, -1, -1};

    // step 1, check the 0-th fft direction
    // either respect the r2c_direction or select a direction where the input grid has pencils format
    if (r2c_direction != -1){
        fft_direction[0] = r2c_direction;
    }else{
        for(int i=0; i<3; i++){ // find a pencil direction
            if (is_pencils(world_in, boxes.in, i)){
                fft_direction[0] = i;
                break;
            }
        }
    }

    // step 2, check the last fft direction
    // must be different from fft_direction[0] and looking to pick direction with pencil format
    for(int i=0; i<3; i++){
        if (i != fft_direction[0] and is_pencils(world_out, boxes.out, i)){
            fft_direction[2] = i;
            break;
        }
    }

    // step 3, make sure that we have a valid direction 0
    // we want the first direction to be different from the final
    if (fft_direction[0] == -1) // if input is non-pencil and not doing r2c
        fft_direction[0] = get_any_valid(fft_direction); // just pick the first unused direction
    // at this point we definitely have a valid direction for fft 0 and maybe have a direction for the last fft

    // step 4, setup the shape for right before fft 0 and right after (take into account the r2c changes)

    // shape 0 comes right before fft 0
    // if the final configuration uses pencils in all directions, just jump to that
    std::vector<box3d> shape0 = next_pencils_shape(world_in, proc_grid, fft_direction[0], boxes.in, opts.use_reorder,
                                                   world_out, {0, 1, 2}, boxes.out);

    // shape fft0 comes right after fft 0
    std::vector<box3d> shape_fft0 = apply_r2c(shape0, r2c_direction);

    // step 5, pick direction for the middle fft
    // must be different from the others and try to pick a direction with existing pencils
    for(int i=0; i<3; i++){
        if (i != fft_direction[0] and i != fft_direction[2] and is_pencils(world_out, shape_fft0, i)){
            fft_direction[1] = i;
            break;
        }
        if (fft_direction[1] == -1) // did not find pencils
            fft_direction[1] = get_any_valid(fft_direction);
    }

    // step 6, make sure we have a final direction
    if (fft_direction[2] == -1){ // if the final configuration is not made of pencils
        fft_direction[2] = get_any_valid(fft_direction);
    }

    std::vector<box3d> shape1 = next_pencils_shape(world_out, proc_grid, fft_direction[1], shape_fft0, opts.use_reorder,
                                                   world_out, {fft_direction[1], fft_direction[2]}, boxes.out);

    std::vector<box3d> shape2 = next_pencils_shape(world_out, proc_grid, fft_direction[2], shape1, opts.use_reorder,
                                                   world_out, {fft_direction[2]}, boxes.out);

    return {
        {boxes.in, shape_fft0, shape1, shape2},
        {shape0,   shape1,     shape2, boxes.out},
        fft_direction,
        world_in.count()
           };
}

}
