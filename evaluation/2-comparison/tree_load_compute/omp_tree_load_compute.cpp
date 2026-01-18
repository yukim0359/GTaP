// omp_tree_load_compute.cpp
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <omp.h>

// ------------------------------
// Helpers: hash + FMA-mix
// ------------------------------
static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline double mix_fma(double x) {
    x = std::fma(x, 1.0000001192092896, 0.9999999403953552);
    x = std::fma(x, 0.9999999403953552, 1.0000001192092896);
    return x;
}

// ------------------------------
// Work per node: mem loads + compute
// (CPU version: single-thread within the task)
// ------------------------------
static inline double do_memory_and_compute_cpu(
    int node,
    int mem_ops,
    int compute_iters,
    const double* input, int input_n,
    const int* indices, int indices_n
) {
    // 1) fixed number of random global loads via index stream
    uint32_t base = hash32((uint32_t)node) % (uint32_t)indices_n;
    double acc = 0.0;

    for (int m = 0; m < mem_ops; ++m) {
        int idx = indices[(base + (uint32_t)m) % (uint32_t)indices_n];
        // idx in [0, input_n)
        acc += input[idx];
    }

    // 2) variable compute: FMA-heavy loop
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = 0; it < compute_iters; ++it) {
        x = mix_fma(x);
    }
    return x;
}

// ------------------------------
// Recursive task function (binary tree)
// ------------------------------
static void tree_work_omp(
    int node, int height, int mem_ops, int compute_iters,
    const double* input, int input_n,
    const int* indices, int indices_n,
    double* out, int total_nodes
) {
    // bounds guard (safety)
    if (node >= total_nodes) return;

    if (height == 0) {
        double v = do_memory_and_compute_cpu(node, mem_ops, compute_iters,
                                             input, input_n, indices, indices_n);
        out[node] = v;
        return;
    }

    int l = node * 2 + 1;
    int r = node * 2 + 2;

    // spawn children
    #pragma omp task default(none) firstprivate(l, height, mem_ops, compute_iters, input, input_n, indices, indices_n, out, total_nodes) \
                     depend(out: out[l])
    {
        tree_work_omp(l, height - 1, mem_ops, compute_iters,
                      input, input_n, indices, indices_n, out, total_nodes);
    }

    #pragma omp task default(none) firstprivate(r, height, mem_ops, compute_iters, input, input_n, indices, indices_n, out, total_nodes) \
                     depend(out: out[r])
    {
        tree_work_omp(r, height - 1, mem_ops, compute_iters,
                      input, input_n, indices, indices_n, out, total_nodes);
    }

    // wait children (explicit + depend to be robust)
    #pragma omp taskwait

    // combine
    double own = do_memory_and_compute_cpu(node, mem_ops, compute_iters,
                                          input, input_n, indices, indices_n);
    double lv = out[l];
    double rv = out[r];
    out[node] = (lv + rv) * 0.5 + own * 1e-6;
}

int main(int argc, char** argv) {
    int height = 15;
    int mem_ops = 64;
    int compute_iters = 512;
    int input_n = 1 << 20;
    int indices_n = 1 << 20;

    if (argc >= 2) height = std::atoi(argv[1]);
    if (argc >= 3) compute_iters = std::atoi(argv[2]);
    if (argc >= 4) mem_ops = std::atoi(argv[3]);

    const int total_nodes = (1 << (height + 1)) - 1;

    std::printf("OMP Tree workload: height=%d nodes=%d mem_ops=%d compute_iters=%d\n",
                height, total_nodes, mem_ops, compute_iters);
    std::printf("OMP max threads: %d\n", omp_get_max_threads());

    // Host init (same spirit as your GPU version)
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<int> idist(0, input_n - 1);

    std::vector<double> input(input_n);
    for (int i = 0; i < input_n; ++i) input[i] = dist(rng);

    std::vector<int> indices(indices_n);
    for (int i = 0; i < indices_n; ++i) indices[i] = idist(rng);

    std::vector<double> out((size_t)total_nodes, 0.0);

    #pragma omp parallel
    {
        #pragma omp single
        {
            /* no op */
        }
    }

    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            tree_work_omp(0, height, mem_ops, compute_iters,
                          input.data(), input_n,
                          indices.data(), indices_n,
                          out.data(), total_nodes);
        }
    }
    double t1 = omp_get_wtime();
    double ms = (t1 - t0) * 1000.0;

    std::printf("Root: %.6e\n", out[0]);
    std::printf("Execution time: %.3f ms\n", ms);

    return 0;
}
