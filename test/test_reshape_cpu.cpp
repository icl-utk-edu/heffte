/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

#ifdef Heffte_ENABLE_FFTW
using default_cpu_backend = heffte::backend::fftw;
#endif

/*
 * Simple unit test that checks the operation that gathers boxes across an mpi comm.
 */
void test_boxes(MPI_Comm const comm){
    current_test<> test("heffte::mpi::gather_boxes", comm);

    int const me = mpi::comm_rank(comm);

    std::vector<box3d> reference_inboxes;
    std::vector<box3d> reference_outboxes;

    for(int i=0; i<mpi::comm_size(comm); i++){
        reference_inboxes.push_back({{i, i+1, i+2}, {i+3, i+4, i+5}});
        reference_outboxes.push_back({{i, i+3, i+5}, {i+7, i+6, i+9}});
    }

    ioboxes boxes = mpi::gather_boxes(reference_inboxes[me], reference_outboxes[me], comm);

    tassert(match(boxes.in,  reference_inboxes));
    tassert(match(boxes.out, reference_outboxes));
}

/*
 * Returns a vector of data corresponding to a sub-box of the original world.
 * The entries are floating point numbers (real or complex) but have integer values
 * corresponding to the indexes in the world box.
 * Thus, by checking the indexes, it is easy to check if data was moved correctly
 * from one sub-box to another.
 */
template<typename scalar_type>
std::vector<scalar_type> get_subdata(box3d const world, box3d const subbox){
    // the entries in the master box go 0, 1, 2, 3, 4 ...
    int const wmidstride  = world.size[0];
    int const wslowstride = world.size[0] * world.size[1];
    int const smidstride  = subbox.size[0];
    int const sslowstride = subbox.size[0] * subbox.size[1];

    std::vector<scalar_type> result(subbox.count());

    for(int k = 0; k < subbox.size[2]; k++){
        for(int j = 0; j < subbox.size[1]; j++){
            for(int i = 0; i < subbox.size[0]; i++){
                result[k * sslowstride + j * smidstride + i]
                    = static_cast<scalar_type>((k + world.low[2] + subbox.low[2]) * wslowstride
                                                + (j + world.low[1] + subbox.low[1]) * wmidstride
                                                + i + world.low[0] + subbox.low[0]);
            }
        }
    }
    return result;
}

// splits the world box into a set of boxes with gird given by proc_grid


template<int hfast, int hmid, int hslow, int pfast, int pmid, int pslow, typename scalar_type>
void test(MPI_Comm const comm){
    /*
     * simple test, create a world of indexes going all the way to hfast, hmid and hslow
     * then split the world into boxes numbering pfast, pmid, and pslow, assume that's what each rank owns
     * then create a new world of pencils and assigns a pencil to each rank (see the shuffle comment)
     * more the data from the original configuration to the new and check against reference data
     */
    current_test<scalar_type> test("heffte::reshape3d_alltoallv -" + std::to_string(mpi::comm_size(comm)), comm);
    tassert( pfast * pmid * pslow == heffte::mpi::comm_size(comm) );

    int const me = heffte::mpi::comm_rank(comm);
    int const shift = 3;

    box3d world = {{0, 0, 0}, {hfast, hmid, hslow}};

    auto boxes   = split_world(world, {pfast, pmid, pslow});
    auto pencils = split_world(world, {pfast,    1, pmid * pslow});

    std::vector<box3d> rotate_boxes;
    if (std::is_same<scalar_type, std::complex<float>>::value){
        // shuffle the pencil boxes in some tests to check the case when there is no overlap between inbox and outbox
        // for the 2 by 2 grid, this shuffle ensures no overlap
        for(size_t i=0; i<boxes.size(); i++) rotate_boxes.push_back( pencils[(i + shift) % boxes.size()] );
    }else{
        for(auto b : pencils) rotate_boxes.push_back(b);
    }

    // create caches for a reshape algorithm, including creating a new mpi comm
    auto reshape = make_reshape3d_alltoallv<default_cpu_backend>(boxes, rotate_boxes, comm);

    auto input_data     = get_subdata<scalar_type>(world, boxes[me]);
    auto reference_data = get_subdata<scalar_type>(world, rotate_boxes[me]);
    auto output_data    = std::vector<scalar_type>(rotate_boxes[me].count());

    if (std::is_same<scalar_type, float>::value){
        // sometimes, run two tests to make sure there is no internal corruption
        // there is no need to do that for every data type
        reshape->apply(input_data.data(), output_data.data());
        output_data = std::vector<scalar_type>(rotate_boxes[me].count());
        reshape->apply(input_data.data(), output_data.data());
    }else{
        reshape->apply(input_data.data(), output_data.data());
    }

    // mpi::dump(0, input_data,     "input");
    // mpi::dump(0, output_data,    "output");
    // mpi::dump(0, reference_data, "reference");

    tassert(match(output_data, reference_data));
}

void perform_tests(){
    all_tests<> name("heffte reshape methods");

    MPI_Comm const comm = MPI_COMM_WORLD;

    test_boxes(comm);

    switch(mpi::comm_size(comm)) {
        // note that the number of boxes must match the comm size
        // that is the product of the last three of the box dimensions
        case 4:
            test<10, 13, 10, 2, 2, 1, float>(comm);
            test<10, 20, 17, 2, 2, 1, double>(comm);
            test<30, 10, 10, 2, 2, 1, std::complex<float>>(comm);
            test<11, 10, 13, 2, 2, 1, std::complex<double>>(comm);
            break;
        case 12:
            test<13, 13, 10, 3, 4, 1, float>(comm);
            test<16, 21, 17, 3, 1, 4, double>(comm);
            test<38, 13, 20, 1, 4, 3, std::complex<float>>(comm);
            test<41, 17, 15, 3, 2, 2, std::complex<double>>(comm);
            break;
        default:
            // unknown test
            break;
    }
}

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    perform_tests();

    MPI_Finalize();

    return 0;
}
