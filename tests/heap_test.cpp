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
const size_t TEST_SIZE = 100000000;

using namespace faiss;
using namespace efanna2e;

int main() {
    std::minstd_rand rng(2023 * 7741 + omp_get_thread_num());
    std::uniform_real_distribution<double> u(0, 255);

    //default heap (std::vector)
    std::vector<Neighbor> heap1[10000];
    for (auto &e: heap1)
        e.reserve(HEAP_SIZE);

    //packaged faiss heap
    std::vector<nhood> heap2(10000);
    for (auto &e: heap2)
        e.init(HEAP_SIZE, 0);

    //simple faiss heap
    HeapArray<CMax<float, std::pair<unsigned, bool>>> heap3[10000];
    for (auto &e: heap3) {
        e.val = new float[HEAP_SIZE];
        e.ids = new std::pair<unsigned, bool>[HEAP_SIZE];
        e.k = 0;
    }


    std::vector<std::pair<unsigned, Neighbor>> query;

    std::cout << "Control the heap size to " << HEAP_SIZE << std::endl;

    //generate the test query set
    std::cout << "Making " << TEST_SIZE << " queries..\n";
    for (size_t i = 0; i < TEST_SIZE; i++) {
        query.push_back(std::make_pair(rng() % 10000, Neighbor(rng() % 10000000, u(rng), true)));
    }

    std::cout << "Make queries done, starting benchmark test..\n";

    { //test for simple faiss heap
        auto s = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_SIZE; i++) {
            unsigned heap_id = query[i].first;
            unsigned id = query[i].second.id;
            float dist = query[i].second.distance;
            auto &e = heap3[heap_id];

            bool flag = false;
            for (int j = 0; j < e.k; j++)
                if (id == e.ids[j].first) {
                    flag = true;
                    break;
                }
            if (!flag) {
                if (e.k < HEAP_SIZE) {
                    e.k += 1;
                    faiss::heap_push<faiss::CMax<float, std::pair<unsigned, bool>>>(e.k, e.val, e.ids, dist,
                                                                                    std::make_pair(id, true));
                } else {
                    faiss::heap_replace_top<faiss::CMax<float, std::pair<unsigned, bool>>>(e.k, e.val,
                                                                                           e.ids,
                                                                                           dist,
                                                                                           std::make_pair(id, true));
                }
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Simple Faiss heap time: " << diff.count() << "\n";
    }

    { //test for faiss heap
        auto s = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_SIZE; i++) {
            unsigned heap_id = query[i].first;
            unsigned id = query[i].second.id;
            float dist = query[i].second.distance;
            auto &e = heap2[heap_id];

            e.insert(id, dist);
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Packaged Faiss heap time: " << diff.count() << "\n";
    }

    { //test for efanna heap
        auto s = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_SIZE; i++) {
            unsigned heap_id = query[i].first;
            unsigned id = query[i].second.id;
            float dist = query[i].second.distance;

            auto &e1 = heap1[heap_id];

            auto it = std::find_if(e1.begin(), e1.end(), [id](Neighbor const &obj) {
                return obj.id == id;
            });
            if (it == e1.end()) {
                if (e1.size() < e1.capacity()) {
                    e1.emplace_back(id, dist, true);
                    std::push_heap(e1.begin(), e1.end());
                } else {
                    std::pop_heap(e1.begin(), e1.end());
                    e1[e1.size() - 1] = Neighbor(id, dist, true);
                    std::push_heap(e1.begin(), e1.end());
                }
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Default heap time: " << diff.count() << "\n";
    }

    return 0;
}