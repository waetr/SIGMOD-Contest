//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#include <faiss/utils/distances.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/exceptions.h>
#include <efanna2e/parameters.h>
#include <omp.h>
#include <set>

namespace efanna2e {
#define _CONTROL_NUM 100

    IndexGraph::IndexGraph(const size_t dimension, const size_t n, Metric m, const size_t l)
            : Index(dimension, n, m) {
        pool_capacity = 100;
        pool = new faiss::HeapArray<faiss::CMax<float, std::pair<unsigned, bool>>>[n];
        for (size_t i = 0; i < n; i++) {
            pool[i].val = new float[l];
            pool[i].ids = new std::pair<unsigned, bool>[l];
            pool[i].k = 0;
            pool[i].nh = 100;
        }
    }

    IndexGraph::~IndexGraph() {
        for (size_t i = 0; i < nd_; i++) {
            delete[] pool[i].val;
            delete[] pool[i].ids;
        }
        delete[] pool;
    }

    void IndexGraph::heap_insert(const size_t n, unsigned id,
                                 float dist,
                                 std::mutex &m) {

        faiss::HeapArray<faiss::CMax<float, std::pair<unsigned, bool>>> &pool_ = pool[n];
        for (int i = 100; i < pool_.nh; i++) {
            if (pool_.ids[i].first == id) return;
        }
        for (int i = 0; i < pool_.k; i++) {
            if (pool_.ids[i].first == id) return;
        }
        LockGuard guard(m);
        if (pool_.k < pool_capacity) {
            pool_.k += 1;
            faiss::heap_push<faiss::CMax<float, std::pair<unsigned, bool>>>(pool_.k, pool_.val, pool_.ids, dist,
                                                                            std::make_pair(id, true));
        } else {
            faiss::heap_replace_top<faiss::CMax<float, std::pair<unsigned, bool>>>(pool_.k, pool_.val,
                                                                                   pool_.ids,
                                                                                   dist,
                                                                                   std::make_pair(id, true));
        }
    }

    void IndexGraph::join() {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (unsigned n = 0; n < nd_; n++) {
            graph_[n].join([&](unsigned i, unsigned j) {
                if (i != j) {
                    float dist = distance_->compare(data_ + i * dimension_, data_ + j * dimension_, dimension_);
                    if (dist <= pool[i].val[0]) {
                        heap_insert(i, j, dist, graph_[n].lock);
                    }
                    if (dist <= pool[j].val[0]) {
                        heap_insert(j, i, dist, graph_[n].lock);
                    }
                }
            });
        }
    }

    void IndexGraph::update(const Parameters &parameters) {
        unsigned S = parameters.Get<unsigned>("S");
        unsigned R = parameters.Get<unsigned>("R");
        unsigned L = parameters.Get<unsigned>("L");
        // Step 1.
        // Clear all nn_new and nn_old
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            std::vector<unsigned>().swap(graph_[i].nn_new);
            std::vector<unsigned>().swap(graph_[i].nn_old);
        }

        int64_t cnt_l = 0, cnt_c = 0;
        // Step 2.
        // Compute the number of neighbors which is new i.e. flag is true
        // in the candidate pool. This must not exceed the sample number S.
        // That means We only select S new neighbors.
#pragma omp parallel for reduction(+:cnt_l), reduction(+:cnt_c)
        for (unsigned n = 0; n < nd_; ++n) {
            faiss::heap_reorder<faiss::CMax<float, std::pair<unsigned, bool>>>
                    (pool[n].k, pool[n].val, pool[n].ids);
            auto &nn = graph_[n];
            unsigned maxl = std::min(nn.M + S, (unsigned) pool[n].k);
            unsigned c = 0;
            unsigned l = 0;
            while ((l < maxl) && (c < S)) {
                if (pool[n].ids[l].second) ++c;
                ++l;
            }
            nn.M = l;
            cnt_l += l;
            cnt_c += c;
        }

        // Step 3.
        // Find reverse links for each node
        // Randomly choose R reverse links.
#pragma omp parallel default(shared)
        {
            std::minstd_rand rng(2023 * 7741 + omp_get_thread_num());
            //std::mutex lock;
#pragma omp for
            for (unsigned n = 0; n < nd_; ++n) {
                auto &nnhd = graph_[n];
                auto &nn_new = nnhd.nn_new;
                auto &nn_old = nnhd.nn_old;
                for (unsigned l = 0; l < nnhd.M; ++l) {
                    auto &nn_distance = pool[n].val[l];
                    auto &nn_ids = pool[n].ids[l];
                    auto &nhood_o = graph_[nn_ids.first];  // nn on the other side of the edge

                    if (nn_ids.second) { // the node is inserted newly
                        // push the neighbor into nn_new
                        nn_new.push_back(nn_ids.first);
                        // push itself into other.rnn_new if it is not in
                        // the candidate pool of the other side
                        if (nn_distance > pool[nn_ids.first].val[pool[nn_ids.first].k - 1]) {
                            LockGuard guard(nhood_o.lock);
                            if (nhood_o.rnn_new.size() < R)nhood_o.rnn_new.push_back(n);
                            else {
                                unsigned int pos = rng() % R;
                                nhood_o.rnn_new[pos] = n;
                            }
                        }
                        nn_ids.second = false;
                    } else { // the node is old
                        // push the neighbor into nn_old
                        nn_old.push_back(nn_ids.first);
                        // push itself into other.rnn_old if it is not in
                        // the candidate pool of the other side
                        if (nn_distance > pool[nn_ids.first].val[pool[nn_ids.first].k - 1]) {
                            LockGuard guard(nhood_o.lock);
                            if (nhood_o.rnn_old.size() < R)nhood_o.rnn_old.push_back(n);
                            else {
                                unsigned int pos = rng() % R;
                                nhood_o.rnn_old[pos] = n;
                            }
                        }
                    }
                }

                for (unsigned l = 100; l < pool[n].nh; ++l) {
                    auto &nn_distance = pool[n].val[l];
                    auto &nn_ids = pool[n].ids[l];
                    auto &nhood_o = graph_[nn_ids.first];  // nn on the other side of the edge
                    assert(nn_ids.second == false);
                    nn_old.push_back(nn_ids.first);
                    // push itself into other.rnn_old if it is not in
                    // the candidate pool of the other side
                    if (nn_distance > pool[nn_ids.first].val[pool[nn_ids.first].k - 1]) {
                        LockGuard guard(nhood_o.lock);
                        if (nhood_o.rnn_old.size() < R)nhood_o.rnn_old.push_back(n);
                        else {
                            unsigned int pos = rng() % R;
                            nhood_o.rnn_old[pos] = n;
                        }
                    }
                }
                // make heap to join later (in join() function)
                if (pool[n].nh < L) {
                    for (unsigned l = 0; l < nnhd.M; ++l) {
                        assert(!pool[n].ids[l].second);
                        pool[n].val[pool[n].nh] = pool[n].val[l];
                        pool[n].ids[pool[n].nh] = pool[n].ids[l];
                        pool[n].nh++;
                        pool[n].val[l] = pool[n].val[pool[n].k - 1];
                        pool[n].ids[l] = pool[n].ids[pool[n].k - 1];
                        pool[n].k--;
                        if (pool[n].nh >= L || l >= pool[n].k) break;
                    }
                }

                faiss::heap_heapify<faiss::CMax<float, std::pair<unsigned, bool>>>(pool[n].k, pool[n].val, pool[n].ids,
                                                                                   pool[n].val, pool[n].ids, pool[n].k);
            }
        }

        // Step 4.
        // Combine the forward and the reverse links
        // R = 0 means no reverse links are used.
#pragma omp parallel
        {
            std::minstd_rand rng(2023 * 7741 + omp_get_thread_num());
#pragma omp for
            for (unsigned i = 0; i < nd_; ++i) {
                auto &nn_new = graph_[i].nn_new;
                auto &nn_old = graph_[i].nn_old;
                auto &rnn_new = graph_[i].rnn_new;
                auto &rnn_old = graph_[i].rnn_old;

                nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());

                nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
                if (nn_old.size() > R) {
                    std::shuffle(nn_old.begin(), nn_old.end(), rng);
                    nn_old.resize(R);
                    nn_old.reserve(R);
                }
                std::vector<unsigned>().swap(graph_[i].rnn_new);
                std::vector<unsigned>().swap(graph_[i].rnn_old);
            }
        }

        printf(" l: %.3f c: %.3f\n", 1.0 * cnt_l / nd_, 1.0 * cnt_c / nd_);
    }

    void IndexGraph::NNDescent(const Parameters &parameters) {
        unsigned iter = parameters.Get<unsigned>("iter");
        std::mt19937 rng(rand());
        std::vector<unsigned> control_points(_CONTROL_NUM);
        std::vector<std::vector<unsigned> > acc_eval_set(_CONTROL_NUM);
        GenRandom(rng, &control_points[0], control_points.size(), nd_);
        generate_control_set(control_points, acc_eval_set, nd_);
        double tot_time = 0;
        for (unsigned it = 1; it <= iter; it++) {
            if (it == 1) {
                eval_recall(control_points, acc_eval_set);
                std::cout << "// Initial graph\n";
            }

            std::cout << "iter: " << it;
            auto s = std::chrono::high_resolution_clock::now();

            update(parameters);

            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            std::cout << " update: " << diff.count() << "|";
            tot_time += diff.count();

            s = std::chrono::high_resolution_clock::now();

            join();

            e = std::chrono::high_resolution_clock::now();
            diff = e - s;
            std::cout << "join: " << diff.count() << " ";
            tot_time += diff.count();
            std::cout << "Total time: " << tot_time << " ";
            eval_recall(control_points, acc_eval_set);

        }
    }

    void IndexGraph::generate_control_set(std::vector<unsigned> &c,
                                          std::vector<std::vector<unsigned> > &v,
                                          unsigned N) {
#pragma omp parallel for
        for (unsigned i = 0; i < c.size(); i++) {
            std::vector<Neighbor> tmp;
            for (unsigned j = 0; j < N; j++) {
                //float dist = faiss::fvec_L2sqr(data_ + c[i] * dimension_, data_ + j * dimension_, dimension_);
                float dist = distance_->compare(data_ + c[i] * dimension_, data_ + j * dimension_, dimension_);
                tmp.push_back(Neighbor(j, dist, true));
            }
            std::partial_sort(tmp.begin(), tmp.begin() + _CONTROL_NUM, tmp.end());
            for (unsigned j = 0; j < _CONTROL_NUM; j++) {
                v[i].push_back(tmp[j].id);
            }
        }
    }

    void
    IndexGraph::eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set) {
        float mean_acc = 0;
        for (unsigned i = 0; i < ctrl_points.size(); i++) {
            float acc = 0;
            auto &g = pool[ctrl_points[i]];
            auto &v = acc_eval_set[i];
            for (unsigned j = 0; j < g.k; j++) {
                for (unsigned k = 0; k < v.size(); k++) {
                    if (g.ids[j].first == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            for (unsigned j = 100; j < g.nh; j++) {
                for (unsigned k = 0; k < v.size(); k++) {
                    if (g.ids[j].first == v[k]) {
                        acc++;
                        break;
                    }
                }
            }
            mean_acc += acc / v.size();
        }
        std::cout << "recall : " << mean_acc / ctrl_points.size() << std::endl;
    }

    void IndexGraph::InitializeGraph_Refine(const Parameters &parameters) {
        assert(final_graph_.size() == nd_);

        const unsigned L = parameters.Get<unsigned>("L");
        const unsigned S = parameters.Get<unsigned>("S");
        graph_.reserve(nd_);
        graph_.resize(nd_);
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            graph_[i].init(L, S);
            auto &ids = final_graph_[i];
            //std::sort(ids.begin(), ids.end());

            size_t K_ = ids.size();

            for (size_t j = 0; j < K_; j++) {
                unsigned id = ids[j];
                if (id == i) continue;
                //float dist = faiss::fvec_L2sqr(data_ + i * dimension_, data_ + id * dimension_, (size_t) dimension_);
                float dist = distance_->compare(data_ + i * dimension_, data_ + id * dimension_, (unsigned) dimension_);
                //graph_[i].pool.emplace_back(id, dist, true);
                pool[i].k++;
                faiss::heap_push<faiss::CMax<float, std::pair<unsigned, bool>>>(pool[i].k, pool[i].val, pool[i].ids,
                                                                                dist,
                                                                                std::make_pair(id, true));
            }
            std::vector<unsigned>().swap(ids);
        }
        CompactGraph().swap(final_graph_);
    }


    void IndexGraph::RefineGraph(const float *data, const Parameters &parameters) {
        data_ = data;
        auto s = std::chrono::high_resolution_clock::now();
        InitializeGraph_Refine(parameters);
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "init time: " << diff.count() << std::endl;

        NNDescent(parameters);

        final_graph_.resize(nd_);
        unsigned K = parameters.Get<unsigned>("K");

        // Store the neighbor link structure into final_graph
        // Clear the old graph
        s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (unsigned i = 0; i < nd_; i++) {
            final_graph_[i].reserve(K);
            final_graph_[i].resize(K);
            assert(pool[i].k == pool_capacity-1);
            pool[i].k = pool[i].nh;
            faiss::heap_heapify<faiss::CMax<float, std::pair<unsigned, bool>>>(pool[i].k, pool[i].val, pool[i].ids,
                                                                               pool[i].val, pool[i].ids, pool[i].k);
            faiss::heap_reorder<faiss::CMax<float, std::pair<unsigned, bool>>>(pool[i].k, pool[i].val, pool[i].ids);

            for (unsigned j = 0; j < K; j++) {
                final_graph_[i][j] = pool[i].ids[j].first;
            }
        }
        std::vector<nhood>().swap(graph_);
        has_built = true;
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "write time: " << diff.count() << std::endl;
    }

    void IndexGraph::Save(const char *filename) {
        std::ofstream out(filename, std::ios::binary | std::ios::out);
        assert(final_graph_.size() == nd_);
        unsigned GK = (unsigned) final_graph_[0].size();
        for (unsigned i = 0; i < nd_; i++) {
            //out.write((char *) &GK, sizeof(unsigned));
            out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
        }
        out.close();
    }

    void IndexGraph::Load(const char *filename) {
        std::ifstream in(filename, std::ios::binary);
        unsigned k = 100;
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        size_t fsize = (size_t) ss;
        size_t num = fsize / 400;
        in.seekg(0, std::ios::beg);

        std::cout << "# of initial KNNGraph: " << num << std::endl;

        final_graph_.resize(num);
        for (size_t i = 0; i < num; i++) {
            final_graph_[i].resize(k);
            final_graph_[i].reserve(k);
            in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
        }
        in.close();
        std::cout << "initial KNNGraph loaded" << std::endl;
    }
}
