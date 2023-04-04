#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <efanna2e/util.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/distance.h>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <cassert>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>

const size_t HEAP_SIZE = 200;
const size_t TEST_SIZE = 1000000000;

using namespace faiss;
using namespace efanna2e;

int main() {
    std::minstd_rand rng(2023 * 7741 + omp_get_thread_num());
    std::uniform_real_distribution<double> u(0, 255);

    std::vector<Neighbor> heap1;
    heap1.reserve(HEAP_SIZE);
    auto *dists = new float[HEAP_SIZE];
    auto *ids = new std::pair<unsigned, bool>[HEAP_SIZE];
    unsigned size_ = 0;
    std::vector<Neighbor> query;

    std::cout << "Control the heap size to " << HEAP_SIZE << std::endl;

    //generate the test query set
    std::cout << "Making " << TEST_SIZE << " queries..\n";
    for (size_t i = 0; i < TEST_SIZE; i++) {
        query.push_back(Neighbor(rng() % 10000, u(rng), true));
    }

    std::cout << "Make queries done, starting benchmark test..\n";

    { //test for efanna heap
        auto s = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_SIZE; i++) {
            unsigned id = query[i].id;
            float dist = query[i].distance;

            if (heap1.size() < heap1.capacity()) {
                heap1.emplace_back(id, dist, true);
                std::push_heap(heap1.begin(), heap1.end());
            } else {
                std::pop_heap(heap1.begin(), heap1.end());
                heap1[heap1.size() - 1] = Neighbor(id, dist, true);
                std::push_heap(heap1.begin(), heap1.end());
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Default heap time: " << diff.count() << "\n";
    }

    { //test for faiss heap
        auto s = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_SIZE; i++) {
            unsigned id = query[i].id;
            float dist = query[i].distance;
            if (size_ < HEAP_SIZE) {
                size_++;
                heap_push<CMax<float, std::pair<unsigned, bool>>>(size_, dists, ids, dist, std::make_pair(id, true));
            } else {
                heap_replace_top<CMax<float, std::pair<unsigned, bool>>>(size_, dists, ids, dist,
                                                                         std::make_pair(id, true));
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Faiss heap time: " << diff.count() << "\n";
    }
    return 0;
}