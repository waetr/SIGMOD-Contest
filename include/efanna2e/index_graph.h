//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_GRAPH_H
#define EFANNA2E_INDEX_GRAPH_H

#include <cstddef>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <faiss/utils/Heap.h>



namespace efanna2e {

    class IndexGraph : public Index {
    public:
        typedef std::vector<nhood> KNNGraph;
        typedef std::vector<std::vector<unsigned> > CompactGraph;

        explicit IndexGraph(const size_t dimension, const size_t n, Metric m, const size_t l);


        virtual ~IndexGraph();

        virtual void Save(const char *filename) override;

        virtual void Load(const char *filename) override;

        void RefineGraph(const float *data, const Parameters &parameters);

        CompactGraph final_graph_;
        KNNGraph graph_;

        faiss::HeapArray<faiss::CMax<float, std::pair<unsigned, bool>>> *pool;


    private:
        size_t pool_capacity;

        void heap_insert(faiss::HeapArray<faiss::CMax<float, std::pair<unsigned, bool>>> &pool_, unsigned id, float dist) const;

        void InitializeGraph_Refine(const Parameters &parameters);

        void NNDescent(const Parameters &parameters);

        void join();

        void update(const Parameters &parameters, int flag);

        void generate_control_set(std::vector<unsigned> &c,
                                  std::vector<std::vector<unsigned> > &v,
                                  unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);

    };

}

#endif //EFANNA2E_INDEX_GRAPH_H
