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

static inline uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static inline double mix_fma(double x) {
    x = std::fma(x, 1.0000001192092896, 0.9999999403953552);
    return x;
}

// ------------------------------
// Work per node: mem loads + compute
// (CPU version: single-thread within the task)
// ------------------------------
static inline double do_memory_and_compute_cpu(
    uint32_t node_id,
    int mem_ops,
    int compute_iters,
    const double* input, int input_n
) {
    double acc = 0.0;

    uint32_t seed = xorshift32(node_id ^ 0x9e3779b9u);
    uint32_t mask = (uint32_t)input_n - 1u; // input_n is power of two

    for (int m = 0; m < mem_ops; ++m) {
        uint32_t r = xorshift32(seed + (uint32_t)m * 747796405u);
        int idx = (int)(r & mask);
        acc += input[idx];
    }

    double y = 0.0;
    uint32_t mix = (0x9e3779b9u ^ node_id) + (uint32_t)acc;

    for (int it = 0; it < compute_iters; ++it) {
        double x = (double)xorshift32(seed + (uint32_t)it * 747796405u);
        y = mix_fma(x);
        uint64_t ybits;
        __builtin_memcpy(&ybits, &y, sizeof(ybits));
        mix ^= (uint32_t)ybits;
    }

    // asm volatile("" :: "r"(mix), "g"(y) : "memory");
    return (double)mix;
}

static inline void sink_store(uint32_t node_id, double x, double* sink, int sink_n) {
    // fixed-size sink (power-of-two recommended)
    uint32_t m = (uint32_t)sink_n - 1u;
    uint32_t pos = hash32(node_id) & m;
    sink[pos] = x; // collisions are OK (last writer wins)
}

// ------------------------------
// Variable-branch tree task (your probability model)
//
// Depth d = D - rem_h (root: d=0, leaf: d=D)
// For a node at depth d, each of its B child candidates is generated
// independently with probability p(d) = rem_h / D = (D - d) / D.
//
// - d=0: rem_h=D => always generate B children
// - d=D: rem_h=0 => leaf
// ------------------------------
static void tree_work_omp(
    uint32_t node_id, int rem_h, int D, int B,
    int mem_ops, int compute_iters,
    const double* input, int input_n,
    double* sink, int sink_n
) {
    // leaf
    if (rem_h == 0) {
        double v = do_memory_and_compute_cpu(node_id, mem_ops, compute_iters, input, input_n);
        sink_store(node_id, v, sink, sink_n);
        return;
    }

    int d = D - rem_h; // depth from root (root: 0)

    // seed: node_id と d から一回だけ作る（hash32 1 回だけ）
    uint32_t s = hash32(node_id ^ (uint32_t)(d * 0x9e3779b9u));

    // fork: consider B child candidates, each spawned independently with prob p
    int spawn = 0;
    for (int k = 0; k < B; ++k) {
        // 乱数更新（超軽量）
        s = xorshift32(s + (uint32_t)k);

        // r in [0, D) を作る（D は小さいので mod で十分）
        // D が 2^n なら r = s & (D-1) にできて更に軽い
        int r = (int)(s % (uint32_t)D);

        if (r < rem_h) {
            uint32_t child = node_id * (uint32_t)B + (uint32_t)(k + 1);
            #pragma omp task default(none) firstprivate(child, rem_h, D, B, mem_ops, compute_iters, input, input_n, sink, sink_n)
            {
                tree_work_omp(child, rem_h - 1, D, B, mem_ops, compute_iters,
                              input, input_n, sink, sink_n);
            }
            spawn++;
        }
    }

    if (spawn > 0) {
        #pragma omp taskwait
    }

    // own work
    double own = do_memory_and_compute_cpu(node_id, mem_ops, compute_iters, input, input_n);
    sink_store(node_id, own, sink, sink_n);
}

int main(int argc, char** argv) {
    int height = 15;
    int B = 3;
    int mem_ops = 64;
    int compute_iters = 512;

    int input_n = 1 << 20;    // power of two
    int sink_n  = 1 << 20;    // power of two (fixed-size)

    if (argc >= 2) height        = std::atoi(argv[1]);
    if (argc >= 3) compute_iters = std::atoi(argv[2]);
    if (argc >= 4) mem_ops       = std::atoi(argv[3]);
    if (argc >= 5) B             = std::atoi(argv[4]);

    std::printf("Tree workload (probabilistic B-ary): height=%d B=%d mem_ops=%d compute_iters=%d\n",
                height, B, mem_ops, compute_iters);
    std::printf("Child spawn prob at depth d: p(d)=1-d/D (d=0 => 1)\n");
    std::printf("input_n=%d sink_n=%d\n", input_n, sink_n);
    std::printf("OMP max threads: %d\n", omp_get_max_threads());

    // Host init
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<double> input(input_n);
    for (int i = 0; i < input_n; ++i) input[i] = dist(rng);

    std::vector<double> sink((size_t)sink_n, 0.0);

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
            tree_work_omp(0u, height, height, B, mem_ops, compute_iters,
                          input.data(), input_n,
                          sink.data(), sink_n);
        }
    }
    double t1 = omp_get_wtime();
    double ms = (t1 - t0) * 1000.0;

    // Read back a few sink entries as a lightweight sanity check
    std::printf("Sink[0..7]: ");
    for (int i = 0; i < 8; ++i) std::printf("%.3e ", sink[i]);
    std::printf("\n");
    std::printf("Execution time: %.3f ms\n", ms);

    return 0;
}
