#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace rectmul_host {

constexpr int tile_edge = 16;
constexpr int tile_elements = tile_edge * tile_edge;
// GTaP block-mode leaf uses this many CUDA threads on one tile. Thread mode
// divides the requested cutoff by this so one thread does not absorb a whole
// tile leaf.
constexpr int leaf_threads = 256;
struct Tile {
    double values[tile_elements];
};

struct Options {
    int x = 64;
    int y = 64;
    int z = 64;
    unsigned long long cutoff = 1;
};

inline bool parse_positive(const char* text, const char* name,
                           unsigned long long* value) {
    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0) {
        std::fprintf(stderr, "invalid %s: %s (must be a positive integer)\n",
                     name, text);
        return false;
    }
    *value = parsed;
    return true;
}

inline bool parse_options(int argc, char** argv, Options* opts) {
    if (argc > 5) {
        std::fprintf(stderr,
                     "usage: %s [x_blocks [y_blocks [z_blocks [cutoff]]]]\n",
                     argv[0]);
        return false;
    }
    unsigned long long x = opts->x, y = opts->y, z = opts->z;
    if ((argc > 1 && !parse_positive(argv[1], "x_blocks", &x)) ||
        (argc > 2 && !parse_positive(argv[2], "y_blocks", &y)) ||
        (argc > 3 && !parse_positive(argv[3], "z_blocks", &z)) ||
        (argc > 4 && !parse_positive(argv[4], "cutoff", &opts->cutoff))) {
        return false;
    }
    if (x > INT_MAX || y > INT_MAX || z > INT_MAX) {
        std::fprintf(stderr, "block dimensions must fit in int\n");
        return false;
    }
    opts->x = static_cast<int>(x);
    opts->y = static_cast<int>(y);
    opts->z = static_cast<int>(z);
    return true;
}

inline void fill(std::vector<Tile>& matrix, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (Tile& tile : matrix)
        for (double& value : tile.values) value = dist(rng);
}

inline void pack_colmajor(const std::vector<Tile>& tiles, int brows, int bcols,
                          int tile_ldb, double* out) {
    const int rows = brows * tile_edge;
    for (int bi = 0; bi < brows; ++bi) {
        for (int bj = 0; bj < bcols; ++bj) {
            const Tile& tile = tiles[static_cast<size_t>(bi) * tile_ldb + bj];
            for (int i = 0; i < tile_edge; ++i) {
                for (int j = 0; j < tile_edge; ++j) {
                    out[(bi * tile_edge + i) + (bj * tile_edge + j) * rows] =
                        tile.values[i * tile_edge + j];
                }
            }
        }
    }
}

inline bool verify(const std::vector<Tile>& a, const std::vector<Tile>& b,
                   const std::vector<Tile>& r, int x, int y, int z) {
    const char* skip_verify = std::getenv("RECTMUL_SKIP_VERIFY");
    if (skip_verify != nullptr && skip_verify[0] != '\0' &&
        !(skip_verify[0] == '0' && skip_verify[1] == '\0')) {
        std::printf("Verification: SKIPPED (RECTMUL_SKIP_VERIFY=%s)\n",
                    skip_verify);
        return true;
    }

    const int m = x * tile_edge;
    const int k = y * tile_edge;
    const int n = z * tile_edge;
    std::vector<double> a_col(static_cast<size_t>(m) * k);
    std::vector<double> b_col(static_cast<size_t>(k) * n);
    std::vector<double> r_col(static_cast<size_t>(m) * n);
    std::vector<double> c_col(static_cast<size_t>(m) * n);
    pack_colmajor(a, x, y, y, a_col.data());
    pack_colmajor(b, y, z, z, b_col.data());
    pack_colmajor(r, x, z, z, r_col.data());

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;
    cublasHandle_t handle = nullptr;
    const auto cleanup = [&]() {
        if (handle != nullptr) cublasDestroy(handle);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    };

    const size_t a_bytes = a_col.size() * sizeof(double);
    const size_t b_bytes = b_col.size() * sizeof(double);
    const size_t c_bytes = c_col.size() * sizeof(double);
    if (cudaMalloc(&d_a, a_bytes) != cudaSuccess ||
        cudaMalloc(&d_b, b_bytes) != cudaSuccess ||
        cudaMalloc(&d_c, c_bytes) != cudaSuccess ||
        cudaMemcpy(d_a, a_col.data(), a_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_b, b_col.data(), b_bytes, cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublas verify setup failed\n");
        cleanup();
        return false;
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    if (cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m,
                    d_b, k, &beta, d_c, m) != CUBLAS_STATUS_SUCCESS ||
        cudaMemcpy(c_col.data(), d_c, c_bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "cublasDgemm failed\n");
        cleanup();
        return false;
    }
    cleanup();

    double max_error = 0.0;
    double max_reference = 0.0;
    for (size_t i = 0; i < c_col.size(); ++i) {
        max_error = std::max(max_error, std::abs(r_col[i] - c_col[i]));
        max_reference = std::max(max_reference, std::abs(c_col[i]));
    }
    const double tolerance = 1.0e-10 * std::max(1.0, max_reference);
    std::printf("Verification: %s (max error %.3e, tolerance %.3e)\n",
                max_error <= tolerance ? "PASS" : "FAIL", max_error, tolerance);
    return max_error <= tolerance;
}

}  // namespace rectmul_host
