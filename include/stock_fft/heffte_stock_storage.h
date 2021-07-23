/*
    -- heFFTe --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_STOCK_STORAGE_H
#define HEFFTE_STOCK_STORAGE_H

#include <queue>

#include "heffte_stock_allocator.h"

namespace heffte {

namespace stock{

template<typename F, int L>
class ComplexStorage {
    private:
    struct vecComp {
        constexpr bool operator()(const complex_vector<F,L> &lhs, const complex_vector<F,L> &rhs) {
            return lhs.size() < rhs.size();
        }
    };

    using queue = std::priority_queue<complex_vector<F,L>,
                                    std::vector<complex_vector<F,L>>,
                                    vecComp>;
    
    queue arena;

    public:
    ComplexStorage() = delete;

    ComplexStorage(size_t N) {
        size_t n = N;
        arena = queue {};
        while(n >>= 1) arena.push(complex_vector<F,L>(n));

    }

    ComplexStorage(ComplexStorage<F,L>&& store): arena(std::move(store.arena)) {}

    ComplexStorage<F,L>& operator=(ComplexStorage<F,L>&& store) {
        arena = std::move(store.arena);
        return *this;
    }

    complex_vector<F,L> get(int N) {
        if(arena.empty()) return complex_vector<F,L>(N);
        complex_vector<F,L> ret = arena.pop();
        ret.reserve(N);
        return ret;
    }

    void restore(complex_vector<F,L> v) {arena.push(v);}
    
};

} // stock

} // heffte
#endif // HEFFTE_STOCK_STORAGE_H