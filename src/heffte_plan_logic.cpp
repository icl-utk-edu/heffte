/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "heffte_plan_logic.h"

namespace heffte {

logic_plan3d plan_operations(ioboxes const &boxes, int r2c_direction){

    box3d const world_in  = find_world(boxes.in);
    box3d const world_out = find_world(boxes.out);

    assert( world_complete(boxes.in,  world_in) );
    assert( world_complete(boxes.out, world_out) );
    if (r2c_direction == -1){
        assert( world_in == world_out );
    }else{
        assert( world_in.r2c(r2c_direction) == world_out );
    }

    // the 2-d grid of pencils
    std::array<int, 2> proc_grid = make_procgrid(static_cast<int>(boxes.in.size()));

    std::array<int, 3> fft_direction = (r2c_direction == -1) ? std::array<int, 3>{0, 1, 2} : std::array<int, 3>{r2c_direction, (r2c_direction + 1) % 3, (r2c_direction + 2) % 3};

    // shape 0 comes right after the first reshape
    std::vector<box3d> shape0 = make_pencils(world_in, proc_grid, fft_direction[0], boxes.in);
    // shape fft0 comes right after the first fft
    std::vector<box3d> shape_fft0 = (r2c_direction == -1) ? shape0 : std::vector<box3d>();
    if (r2c_direction != -1)
        for(auto b : shape0)
            shape_fft0.push_back(b.r2c(r2c_direction));

    std::vector<box3d> shape1 = make_pencils(world_out, proc_grid, fft_direction[1], shape_fft0);
    std::vector<box3d> shape2 = make_pencils(world_out, proc_grid, fft_direction[2], shape1);

    return {
        {boxes.in, shape_fft0, shape1, shape2},
        {shape0,   shape1,     shape2, boxes.out},
        fft_direction,
        world_in.count()
           };
}

}
