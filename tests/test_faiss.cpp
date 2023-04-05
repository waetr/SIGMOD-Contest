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

const int K = 100;
const char *index_file = "rep/IVF32768_HNSW32.index";

//const char *search_key = "nprobe=8,quantizer_efSearch=128"; // 0.235 recall within 200 secs
const char *search_key = "nprobe=16,quantizer_efSearch=128"; // 0.307 recall within 270 secs
//const char *search_key = "nprobe=32,quantizer_efSearch=128"; // 0.312 recall within 380 secs (bad)

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << argc << argv[0] << "data_file saving_graph L iter S R" << std::endl;
        exit(-1);
    }
    const unsigned L = atoi(argv[3]);
    const unsigned iter = atoi(argv[4]);
    const unsigned S = atoi(argv[5]);
    const unsigned R = atoi(argv[6]);
    //omp_set_num_threads(64);
    auto s_ = std::chrono::high_resolution_clock::now();

    float *data_load;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);

    faiss::Index *index = faiss::read_index(index_file);
    assert(index->is_trained);

    {
        std::cout << "Adding...\n";
        auto s = std::chrono::high_resolution_clock::now();

        index->add(points_num, data_load);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "Adding time cost: " << diff.count() << "\n";
    }

    faiss::ParameterSpace params;
    params.set_index_parameters(index, search_key);

    // output buffers
    const int batch_size = points_num < 1000000 ? points_num : 1000000;
    faiss::idx_t *I = new faiss::idx_t[batch_size * (K + 1)];
    float *D = new float[batch_size * (K + 1)];

    efanna2e::IndexGraph index_graph(104, points_num, efanna2e::L2);

    {
        double search_time = 0;
        index_graph.final_graph_.resize(points_num);
        for (size_t i = 0; i < points_num; i++) {
            index_graph.final_graph_[i].reserve(K + 1);
            index_graph.final_graph_[i].resize(K + 1);
        }

        for (unsigned i = 0; i < points_num / batch_size; i++) {
            auto s = std::chrono::high_resolution_clock::now();
            index->search(batch_size, data_load + i * batch_size * dim, K + 1, D, I);
            for (int j = 0; j < batch_size; j++) {
                for (int k0 = 0; k0 < K + 1; k0++) {
                    index_graph.final_graph_[i * batch_size + j][k0] = *(I + j * (K + 1) + k0);
                }
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            search_time += diff.count();
            std::cout << i + 1 << "/" << points_num / batch_size << " Complete! Time: " << diff.count() << "\n";
        }
        std::cout << "Search time: " << search_time << "\n";
    }

    if (points_num == 10000) {
        index_graph.Save(argv[2]);
        return 0;
    }

    delete index;
    index = nullptr;

    data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    {
        auto s = std::chrono::high_resolution_clock::now();

        index_graph.RefineGraph(data_load, paras);

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        std::cout << "NN-Descent Time cost: " << diff.count() << "\n";
    }
    index_graph.Save(argv[2]);

    auto e_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_ = e_ - s_;
    std::cout << "Total time: " << diff_.count() << std::endl;

    return 0;
}
