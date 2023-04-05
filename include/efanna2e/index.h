//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#ifndef EFANNA2E_INDEX_H
#define EFANNA2E_INDEX_H

#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include "distance.h"
#include "parameters.h"

namespace efanna2e {

    class Index {
    public:
        explicit Index(const size_t dimension, const size_t n, Metric metric);


        virtual ~Index();

        virtual void Save(const char *filename) = 0;

        virtual void Load(const char *filename) = 0;

        inline bool HasBuilt() const { return has_built; }

        inline size_t GetDimension() const { return dimension_; };

        inline size_t GetSizeOfDataset() const { return nd_; }

        inline const float *GetDataset() const { return data_; }

        const float *data_;
        Distance *distance_;
    protected:
        const size_t dimension_;
        size_t nd_;
        bool has_built;
    };

}

#endif //EFANNA2E_INDEX_H
