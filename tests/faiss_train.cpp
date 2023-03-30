//
// Created by 付聪 on 2017/6/21.
//

#include <faiss/index_io.h>
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

const int K = 100;


void load_data(char *filename, float *&data, unsigned &num, unsigned &dim) {// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &num, 4);
    dim = 100;
    std::cout << "# of points: " << num << std::endl;
    data = new float[num * dim * sizeof(float)];
    in.seekg(4, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.read((char *) (data + i * dim), dim * sizeof(float));
    }
    in.close();
}


void
compute_gt_for_tune(const float *data, const float *q, const unsigned nq, const unsigned k, unsigned *gt, unsigned nd) {
    auto distance = new efanna2e::DistanceL2();
    unsigned dim = 100;
#pragma omp parallel for
    for (unsigned i = 0; i < nq; i++) {
        std::vector<efanna2e::Neighbor> res;
        for (unsigned j = 0; j < nd; j++) {
            float dist = distance->compare(q + i * dim, data + j * dim, dim);
            res.emplace_back(j, dist, true);
        }
        std::partial_sort(res.begin(), res.begin() + k, res.end());
        for (unsigned j = 0; j < k; j++) {
            gt[i * k + j] = res[j].id;
        }
    }
}

void AutoTune(float *data_load, unsigned points_num, unsigned dim, faiss::Index *&index) {
    unsigned sample_num = 100;
    float *sample_queries = new float[dim * sample_num];
    std::vector<unsigned> tmp(sample_num);
    std::mt19937 rng;
    efanna2e::GenRandom(rng, tmp.data(), sample_num, points_num);
    for (unsigned i = 0; i < tmp.size(); i++) {
        unsigned id = tmp[i];
        memcpy(sample_queries + i * dim, data_load + id * dim, dim * sizeof(float));
    }
    unsigned k = 10;
    int64_t *gt = new int64_t[k * sample_num];//ground truth
    unsigned *gt_c = new unsigned[k * sample_num];
    compute_gt_for_tune(data_load, sample_queries, sample_num, k, gt_c, points_num);
    for (unsigned i = 0; i < k * sample_num; i++) {
        gt[i] = gt_c[i];
    }
    delete[] gt_c;
    std::string selected_params;
    faiss::IntersectionCriterion crit(sample_num, k);
    crit.set_groundtruth(k, nullptr, gt);
    crit.nnn = k; // by default, the criterion will request only 1 NN

    std::cout << "Preparing auto-tune parameters\n";

    faiss::ParameterSpace params;
    params.initialize(index);
    faiss::OperatingPoints ops;
    params.explore(index, sample_num, sample_queries, crit, &ops);
    for (size_t i = 0; i < ops.optimal_pts.size(); i++) {
        if (ops.optimal_pts[i].perf > 0.5) {
            selected_params = ops.optimal_pts[i].key;
            break;
        }
    }
    std::cout << "best parameters auto-tuned: " << selected_params << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << argc << argv[0] << " train/add data_file index_file saving_graph" << std::endl;
        exit(-1);
    }

    //omp_set_num_threads(64);

    auto s_ = std::chrono::high_resolution_clock::now();

    const char *index_key = "IVF32768_HNSW8";
    const char *search_key = "nprobe=8,quantizer_efSearch=128";
    //"nprobe=2,quantizer_efSearch=32";

    float *data_load;
    unsigned points_num, dim;
    load_data(argv[2], data_load, points_num, dim);

    std::string arg_1 = argv[1];
    if (arg_1 == "train") {
        faiss::Index *index = faiss::index_factory(dim, index_key);

        std::cout << "Training...\n";
        auto s = std::chrono::high_resolution_clock::now();

        index->train(points_num, data_load);
        delete[] data_load;

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Training time cost: " << diff.count() << "\n";

        std::cout << "Saving index to " << argv[3] << std::endl;
        faiss::write_index(index, argv[3]);
    } else if (arg_1 == "add") {

        faiss::Index *index = faiss::read_index(argv[3]);
        assert(index->is_trained);

        std::cout << "Adding...\n";
        auto s = std::chrono::high_resolution_clock::now();
        index->add(points_num, data_load);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Adding time cost: " << diff.count() << "\n";

        //Display the auto-tuned parameters
        //AutoTune(data_load, points_num, dim, index);

        std::cout << "current using parameters: " << search_key << std::endl;
        faiss::ParameterSpace params;
        params.set_index_parameters(index, search_key);

        // output buffers
        const int batch_size = points_num < 100000 ? points_num : 100000;
        faiss::idx_t *I = new faiss::idx_t[batch_size * K];
        float *D = new float[batch_size * K];

        efanna2e::IndexRandom init_index(104, points_num);
        efanna2e::IndexGraph index_graph(104, points_num, efanna2e::L2, (efanna2e::Index *) (&init_index));

        index_graph.final_graph_.resize(points_num);
        for (size_t i = 0; i < points_num; i++) {
            index_graph.final_graph_[i].resize(K);
        }

        for (unsigned i = 0; i < points_num / batch_size; i++) {
            s = std::chrono::high_resolution_clock::now();
            index->search(batch_size, data_load + i * batch_size * dim, K, D, I);
            for (int j = 0; j < batch_size; j++) {
                for (int k0 = 0; k0 < K; k0++) {
                    index_graph.final_graph_[i * batch_size + j][k0] = *(I + j * K + k0);
                }
            }
            e = std::chrono::high_resolution_clock::now();
            diff = e - s;
            std::cout << i + 1 << "/" << points_num / batch_size << " Complete! Time: " << diff.count() << "\n";
        }

        if(points_num == 10000) {
            index_graph.Save(argv[4]);
            return 0;
        }

        data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
        efanna2e::Parameters paras;
        paras.Set<unsigned>("K", K);
        paras.Set<unsigned>("L", 200);
        paras.Set<unsigned>("iter", 8);
        paras.Set<unsigned>("S", 20);
        paras.Set<unsigned>("R", 300);

        s = std::chrono::high_resolution_clock::now();

        index_graph.RefineGraph(data_load, paras);

        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        std::cout << "NN-Descent Time cost: " << diff.count() << "\n";
        index_graph.Save(argv[4]);
    } else {
        std::cout << "argv[1] must be \"train\" or \"add\"!\n";
    }

    auto e_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e_ - s_;
    std::cout << "Total time: " << diff.count() << std::endl;

    return 0;
}
