
#include <efanna2e/index_graph.h>
#include <efanna2e/util.h>
#include <omp.h>

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

int main(int argc, char **argv) {
    if (argc != 8) {
        std::cout << argv[0] << " data_file save_graph K L iter S R" << std::endl;
        exit(-1);
    }
    omp_set_num_threads(64);
    float *data_load = NULL;
    unsigned points_num, dim;
    load_data(argv[1], data_load, points_num, dim);
    char *graph_filename = argv[2];
    unsigned K = (unsigned) atoi(argv[3]);
    unsigned L = (unsigned) atoi(argv[4]);
    unsigned iter = (unsigned) atoi(argv[5]);
    unsigned S = (unsigned) atoi(argv[6]);
    unsigned R = (unsigned) atoi(argv[7]);
    data_load = efanna2e::data_align(data_load, points_num, dim);//one must align the data before build
    efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, L);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("K", K);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("iter", iter);
    paras.Set<unsigned>("S", S);
    paras.Set<unsigned>("R", R);

    index.final_graph_.resize(points_num);
    std::mt19937 rng(2023 * 7741);
    for (int i = 0; i < points_num; i++) {
        index.final_graph_[i].resize(K);
        index.final_graph_[i].reserve(K);
        efanna2e::GenRandom(rng, index.final_graph_[i].data(), K, points_num);
    }
    std::cout << "Random initial KNNG has constructed!\n";

    auto s = std::chrono::high_resolution_clock::now();

    index.RefineGraph(data_load, paras);


    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "Time cost: " << diff.count() << "\n";

    index.Save(graph_filename);
    return 0;
}
