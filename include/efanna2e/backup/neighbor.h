//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_GRAPH_H
#define EFANNA2E_GRAPH_H

#include <cstddef>
#include <vector>
#include <mutex>
#include <unordered_set>
#include <bitset>
#include <faiss/utils/Heap.h>

namespace efanna2e {

    struct Neighbor {
        unsigned id;
        float distance;
        bool flag;

        Neighbor() = default;

        Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}

        inline bool operator<(const Neighbor &other) const {
            return distance < other.distance;
        }
    };

    typedef std::lock_guard<std::mutex> LockGuard;

    struct nhood {
        std::mutex lock;
        unsigned M;

        float *pool_dist;
        std::pair<unsigned, bool> *pool_val;
        size_t pool_size;
        size_t pool_capacity;
        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;
        std::vector<unsigned> rnn_old;
        std::vector<unsigned> rnn_new;

        nhood() {}

        nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
            M = s;
            nn_new.resize(s * 2);
            GenRandom(rng, &nn_new[0], (unsigned) nn_new.size(), N);
            nn_new.reserve(s * 2);
        }

        nhood(const nhood &other) {
            M = other.M;
            std::copy(other.nn_new.begin(), other.nn_new.end(), std::back_inserter(nn_new));
            nn_new.reserve(other.nn_new.capacity());
        }

        void init(unsigned l, unsigned s) {
            M = s;
            pool_dist = new float[l];
            pool_val = new std::pair<unsigned, bool>[l];
            pool_capacity = l;
            pool_size = 0;
        }

        void insert(unsigned id, float dist) {
            LockGuard guard(lock);
            for (int i = 0; i < pool_size; i++)
                if (id == (pool_val + i)->first) return;
            if (pool_size < pool_capacity) {
                pool_size += 1;
                faiss::heap_push<faiss::CMax<float, std::pair<unsigned, bool>>>(pool_size, pool_dist, pool_val, dist,
                                                                                std::make_pair(id, true));
            } else {
                faiss::heap_replace_top<faiss::CMax<float, std::pair<unsigned, bool>>>(pool_size, pool_dist, pool_val,
                                                                                       dist,
                                                                                       std::make_pair(id, true));
            }
        }

        template<typename C>
        void join(C callback) const {
            for (unsigned const i: nn_new) {
                for (unsigned const j: nn_new) {
                    if (i < j) {
                        callback(i, j);
                    }
                }
                for (unsigned const j: nn_old) {
                    callback(i, j);
                }
            }
        }
    };

    struct LockNeighbor {
        std::mutex lock;
        std::vector<Neighbor> pool;
    };

    static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
        // find the location to insert
        int left = 0, right = K - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            addr[K] = nn;
            return K;
        }
        while (left < right - 1) {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)right = mid;
            else left = mid;
        }
        //check equal ID

        while (left > 0) {
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return K + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id)return K + 1;
        memmove((char *) &addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }

}

#endif //EFANNA2E_GRAPH_H
