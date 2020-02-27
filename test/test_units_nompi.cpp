/** @class */
/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#include "test_common.h"

void test_factorize(){
    current_test<int, using_nompi> name("prime factorize");

    std::vector<std::array<int, 2>> reference = {{1, 935}, {5, 187}, {11, 85}, {17, 55}, {55, 17}, {85, 11}, {187, 5}};

    auto factors = heffte::get_factors(935);

    sassert(match(factors, reference));

    reference = {{1, 27}, {3, 9}, {9, 3}};
    factors = heffte::get_factors(reference.front()[1]);
    sassert(match(factors, reference));
}

void test_process_grid(){
    current_test<int, using_nompi> name("process grid");

    std::array<int, 2> reference = {4, 5};
    std::array<int, 2> result = heffte::make_procgrid(20);
    sassert(reference == result);

    reference = {1, 17};
    result = heffte::make_procgrid(17);
    sassert(reference == result);

    reference = {81, 81};
    result = heffte::make_procgrid(6561);
    sassert(reference == result);

    reference = {17, 19};
    result = heffte::make_procgrid(323);
    sassert(reference == result);

    reference = {8, 16};
    result = heffte::make_procgrid(128);
    sassert(reference == result);
}

void test_split_pencils(){
    using namespace heffte;
    current_test<int, using_nompi> name("split pencils");

    box3d world = {{0, 0, 0}, {1, 3, 5}};
    std::vector<box3d> reference = {{{0, 0, 0}, {0, 1, 5}}, {{0, 2, 0}, {0, 3, 5}},
                                    {{1, 0, 0}, {1, 1, 5}}, {{1, 2, 0}, {1, 3, 5}}};
    // note that the order of the boxes moves fastest in the mid-dimension
    // this tests the reordering
    std::vector<box3d> result = make_pencils(world, {2, 2}, 2, reference);
    sassert(match(result, reference));

    std::vector<box3d> reference2 = {{{0, 0, 0}, {1, 1, 2}}, {{0, 2, 0}, {1, 3, 2}},
                                     {{0, 0, 3}, {1, 1, 5}}, {{0, 2, 3}, {1, 3, 5}}};
    std::vector<box3d> result2 = make_pencils(world, {2, 2}, 0, reference);
    sassert(match(result2, reference2));
}


int main(int argc, char *argv[]){

    all_tests<using_nompi> name("Non-MPI Tests");

    test_factorize();
    test_process_grid();
    test_split_pencils();

    return 0;
}
