#include <queue>

#include "heffte_stock_allocator.h"

namespace heffte {

namespace stock{

template<typename F, int L>
class ComplexStorage {
    private:
    using queue = std::priority_queue<complex_vector<F,L>,
                                    std::vector<complex_vector<F,L>>,
                                    vecComp>;
    struct vecComp {
        constexpr bool operator()(const complex_vector<F,L> &lhs, const complex_vector<F,L> &rhs) {
            return lhs.size() < rhs.size();
        }
    };
    queue arena;

    public:
    ComplexStorage() = delete;

    ComplexStorage(size_t N) {
        size_t n = N;
        arena = queue {};
        while(n >>= 1) arena.push(complex_vector<F,L>(n));

    }

    complex_vector<F,L> get(int N) {
        if(arena.empty()) return complex_vector<F,L>(N);
        complex_vector<F,L> ret = arena.pop();
        ret.reserve(N);
        return ret;
    }

    void restore(complex_vector<F,L> v) {arena.push(v);}
    
}

}

}