#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <omp.h>
#include <cmath>

// ------------------------------
// Helpers: FMA-mix
// ------------------------------
static inline double mix_fma(double x) {
    x = std::fma(x, 1.0000001192092896, 0.9999999403953552);
    return x;
}

// ------------------------------
// Work per node: mem loads + compute
// (CPU version: single-thread within the task)
// Sequential memory access variant
// ------------------------------
static inline double do_memory_and_compute_cpu_seqmem(
    int node,
    int mem_ops,
    int compute_iters,
    const double* input, int input_n
) {
    double acc = 0.0;

    // If input_n is power-of-two (default 1<<20), masking is valid and fast.
    uint32_t mask = (uint32_t)input_n - 1u;

    // Sequential access region per node
    uint32_t base = ((uint32_t)node * (uint32_t)mem_ops) & mask;

    // 1) fixed number of sequential loads from input
    for (int m = 0; m < mem_ops; ++m) {
        int idx = (int)((base + (uint32_t)m) & mask);
        acc += input[idx];
    }

    // 2) compute loop (FMA-heavy)
    double x = acc + (double)(node & 0xFF) * 1e-9;
    for (int it = 0; it < compute_iters; ++it) {
        x = mix_fma(x);
    }

    return x;
}

// ------------------------------
// Recursive task function (binary tree)
// ------------------------------
static void tree_work_omp_seqmem(
    int node, int height, int mem_ops, int compute_iters,
    const double* input, int input_n,
    double* out, int total_nodes
) {
    // bounds guard (safety)
    if (node >= total_nodes) return;

    if (height == 0) {
        double v = do_memory_and_compute_cpu_seqmem(node, mem_ops, compute_iters, input, input_n);
        out[node] = v;
        return;
    }

    int l = node * 2 + 1;
    int r = node * 2 + 2;

    // spawn children
    #pragma omp task default(none) firstprivate(l, height, mem_ops, compute_iters, input, input_n, out, total_nodes) \
                     depend(out: out[l])
    {
        tree_work_omp_seqmem(l, height - 1, mem_ops, compute_iters,
                             input, input_n, out, total_nodes);
    }

    #pragma omp task default(none) firstprivate(r, height, mem_ops, compute_iters, input, input_n, out, total_nodes) \
                     depend(out: out[r])
    {
        tree_work_omp_seqmem(r, height - 1, mem_ops, compute_iters,
                             input, input_n, out, total_nodes);
    }

    // wait children (explicit + depend to be robust)
    #pragma omp taskwait

    // own synthetic work only (no reduction of children)
    double own = do_memory_and_compute_cpu_seqmem(node, mem_ops, compute_iters, input, input_n);
    out[node] = own;
}

int main(int argc, char** argv) {
    int height = 15;
    int mem_ops = 64;
    int compute_iters = 512;
    int input_n = 1 << 20;

    if (argc >= 2) height = std::atoi(argv[1]);
    if (argc >= 3) compute_iters = std::atoi(argv[2]);
    if (argc >= 4) mem_ops = std::atoi(argv[3]);

    const int total_nodes = (1 << (height + 1)) - 1;

    std::printf("OMP Tree workload (SEQ mem): height=%d nodes=%d mem_ops=%d compute_iters=%d\n",
                height, total_nodes, mem_ops, compute_iters);
    std::printf("OMP max threads: %d\n", omp_get_max_threads());

    // Host init
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> input(input_n);
    for (int i = 0; i < input_n; ++i) input[i] = dist(rng);

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
            tree_work_omp_seqmem(0, height, mem_ops, compute_iters,
                                 input.data(), input_n,
                                 out.data(), total_nodes);
        }
    }
    double t1 = omp_get_wtime();
    double ms = (t1 - t0) * 1000.0;

    std::printf("Root: %.6e\n", out[0]);
    std::printf("Execution time: %.3f ms\n", ms);

    return 0;
}
