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

logic_plan3d plan_operations(ioboxes const &boxes, int r2c_direction){

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
    std::vector<box3d> shape0 = (is_pencils(world_out, boxes.out, {0, 1, 2})) ? boxes.out :
                               make_pencils(world_in, proc_grid, fft_direction[0], boxes.in);

    // shape fft0 comes right after fft 0
    std::vector<box3d> shape_fft0 = (r2c_direction == -1) ? shape0 : std::vector<box3d>();
    if (r2c_direction != -1)
        for(auto b : shape0)
            shape_fft0.push_back(b.r2c(r2c_direction));

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

    std::vector<box3d> shape1 = (is_pencils(world_out, boxes.out, {fft_direction[1], fft_direction[2]})) ? boxes.out :
                                make_pencils(world_out, proc_grid, fft_direction[1], shape_fft0);

    std::vector<box3d> shape2 = (is_pencils(world_out, boxes.out, fft_direction[2])) ? boxes.out :
                                 make_pencils(world_out, proc_grid, fft_direction[2], shape1);

    return {
        {boxes.in, shape_fft0, shape1, shape2},
        {shape0,   shape1,     shape2, boxes.out},
        fft_direction,
        world_in.count()
           };
}

}
