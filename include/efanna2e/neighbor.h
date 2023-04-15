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

        void init(unsigned l, unsigned s, unsigned r) {
            M = s;
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

}

#endif //EFANNA2E_GRAPH_H
