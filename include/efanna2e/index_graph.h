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


namespace efanna2e {

    class IndexGraph : public Index {
    public:
        typedef std::vector<nhood> KNNGraph;
        typedef std::vector<std::vector<unsigned> > CompactGraph;
        typedef std::vector<LockNeighbor> LockGraph;

        explicit IndexGraph(const size_t dimension, const size_t n, Metric m);


        virtual ~IndexGraph();

        virtual void Save(const char *filename) override;

        virtual void Load(const char *filename) override;

        void RefineGraph(const float *data, const Parameters &parameters);

        CompactGraph final_graph_;
        KNNGraph graph_;


    private:
        void InitializeGraph_Refine(const Parameters &parameters);

        void NNDescent(const Parameters &parameters);

        void join();

        void update(const Parameters &parameters);

        void generate_control_set(std::vector<unsigned> &c,
                                  std::vector<std::vector<unsigned> > &v,
                                  unsigned N);

        void eval_recall(std::vector<unsigned> &ctrl_points, std::vector<std::vector<unsigned> > &acc_eval_set);

    };

}

#endif //EFANNA2E_INDEX_GRAPH_H
