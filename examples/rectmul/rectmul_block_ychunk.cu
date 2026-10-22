#include <climits>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include "gtap_block.cuh"
#include "rectmul_host.hpp"

static constexpr int threads_per_block = rectmul_host::leaf_threads;
static constexpr int tile_elements = rectmul_host::tile_elements;
static_assert(tile_elements % threads_per_block == 0,
              "tile size must be divisible by the thread-block size");

using Tile = rectmul_host::Tile;

// One output tile. A and B are staged in shared memory. Each thread owns a
// strided subset of the tile elements.
__device__ void multiply_output_tile(const Tile* a, const Tile* b, Tile* r,
                                     int y, int ldb, bool add) {
    constexpr int elements_per_thread = tile_elements / threads_per_block;
    __shared__ double a_tile[tile_elements];
    __shared__ double b_tile[tile_elements];

    const int tid = static_cast<int>(threadIdx.x);
    double sums[elements_per_thread];
#pragma unroll
    for (int item = 0; item < elements_per_thread; ++item) {
        const int element = tid + item * threads_per_block;
        sums[item] = add ? r->values[element] : 0.0;
    }

    for (int bk = 0; bk < y; ++bk) {
#pragma unroll
        for (int item = 0; item < elements_per_thread; ++item) {
            const int element = tid + item * threads_per_block;
            a_tile[element] = (a + bk)->values[element];
            b_tile[element] = (b + bk * ldb)->values[element];
        }
        __syncthreads();

#pragma unroll
        for (int item = 0; item < elements_per_thread; ++item) {
            const int element = tid + item * threads_per_block;
            const int row = element / rectmul_host::tile_edge;
            const int col = element % rectmul_host::tile_edge;
#pragma unroll
            for (int k = 0; k < rectmul_host::tile_edge; ++k) {
                sums[item] = fma(a_tile[row * rectmul_host::tile_edge + k],
                                 b_tile[k * rectmul_host::tile_edge + col],
                                 sums[item]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < elements_per_thread; ++item) {
        const int element = tid + item * threads_per_block;
        r->values[element] = sums[item];
    }
}

__device__ void multiply_region(const Tile* a, const Tile* b, Tile* r,
                                int x, int y, int z, int lda, int ldb, int ldr,
                                bool add) {
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < z; ++j) {
            multiply_output_tile(a + i * lda, b + j, r + i * ldr + j, y, ldb,
                                 add);
        }
    }
}

__device__ __forceinline__ int next_y_chunk(int total, int cursor, int limit) {
    int begin = 0;
    int length = total;
    while (length > limit) {
        int left = length / 2;
        if (cursor < begin + left) {
            length = left;
        } else {
            begin += left;
            length -= left;
        }
    }
    return length;
}

__device__ __forceinline__ int y_chunk_limit(int x, int z,
                                             unsigned long long cutoff) {
    int limit = x > z ? x : z;
    const unsigned long long output_tiles =
        static_cast<unsigned long long>(x) * z;
    const unsigned long long cutoff_y = cutoff / output_tiles;
    if (cutoff_y > static_cast<unsigned long long>(limit)) {
        limit = cutoff_y > static_cast<unsigned long long>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(cutoff_y);
    }
    return limit;
}

#pragma gtap function
__device__ void rectmul_task(const Tile* a, const Tile* b, Tile* r, int x,
                             int y, int z, int lda, int ldb, int ldr, bool add,
                             unsigned long long cutoff) {
    const Tile* y_base_a = a;
    const Tile* y_base_b = b;
    int y_total = y;
    int y_cursor = 0;
    bool traversing_y = false;

    while (true) {
        unsigned long long tile_products =
            static_cast<unsigned long long>(x) * y * z;
        if (tile_products <= cutoff) {
            multiply_region(a, b, r, x, y, z, lda, ldb, ldr, add);
        } else if (x >= y && x >= z && x > 1) {
            int x0 = x / 2;
            if (threadIdx.x == 0) {
#pragma gtap task
                rectmul_task(a, b, r, x0, y, z, lda, ldb, ldr, add, cutoff);
#pragma gtap task
                rectmul_task(a + x0 * lda, b, r + x0 * ldr, x - x0, y, z, lda,
                             ldb, ldr, add, cutoff);
            }
#pragma gtap taskwait
        } else if (z >= y && z > 1) {
            int z0 = z / 2;
            if (threadIdx.x == 0) {
#pragma gtap task
                rectmul_task(a, b, r, x, y, z0, lda, ldb, ldr, add, cutoff);
#pragma gtap task
                rectmul_task(a, b + z0, r + z0, x, y, z - z0, lda, ldb, ldr,
                             add, cutoff);
            }
#pragma gtap taskwait
        } else if (y > 1) {
            y_base_a = a;
            y_base_b = b;
            y_total = y;
            y_cursor = 0;
            traversing_y = true;
            y = next_y_chunk(y_total, y_cursor, y_chunk_limit(x, z, cutoff));
            continue;
        } else {
            multiply_region(a, b, r, x, y, z, lda, ldb, ldr, add);
        }

        if (!traversing_y) return;
        y_cursor += y;
        if (y_cursor >= y_total) return;
        y = next_y_chunk(y_total, y_cursor, y_chunk_limit(x, z, cutoff));
        a = y_base_a + y_cursor;
        b = y_base_b + y_cursor * ldb;
        add = true;
    }
}

__global__ void exec_kernel(const Tile* a, const Tile* b, Tile* r, int x,
                            int y, int z, unsigned long long cutoff) {
#pragma gtap entry
    rectmul_task(a, b, r, x, y, z, y, z, z, false, cutoff);
}

int main(int argc, char** argv) {
    rectmul_host::Options opts;
    if (!rectmul_host::parse_options(argc, argv, &opts)) return 2;
    std::printf("Rectmul block ychunk: A=%dx%d blocks, B=%dx%d blocks, "
                "R=%dx%d blocks, tile=%dx%d, cutoff=%llu\n",
                opts.x, opts.y, opts.y, opts.z, opts.x, opts.z,
                rectmul_host::tile_edge, rectmul_host::tile_edge,
                opts.cutoff);

    const size_t a_count = static_cast<size_t>(opts.x) * opts.y;
    const size_t b_count = static_cast<size_t>(opts.y) * opts.z;
    const size_t r_count = static_cast<size_t>(opts.x) * opts.z;
    std::vector<rectmul_host::Tile> a(a_count), b(b_count), r(r_count);
    std::mt19937_64 rng(0x726563746d756cULL);
    rectmul_host::fill(a, rng);
    rectmul_host::fill(b, rng);

    gtap_block_config config{
        .grid_size = 512,
        .block_size = threads_per_block,
        .max_tasks_per_block = 10000,
    };
    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) return 1;

    Tile* d_a = nullptr;
    Tile* d_b = nullptr;
    Tile* d_r = nullptr;
    cudaMalloc(&d_a, a_count * sizeof(Tile));
    cudaMalloc(&d_b, b_count * sizeof(Tile));
    cudaMalloc(&d_r, r_count * sizeof(Tile));
    cudaMemcpy(d_a, a.data(), a_count * sizeof(Tile), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b_count * sizeof(Tile), cudaMemcpyHostToDevice);
    cudaMemset(d_r, 0, r_count * sizeof(Tile));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, d_a, d_b, d_r, opts.x, opts.y, opts.z,
                      opts.cutoff);
    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        err = gtap_synchronize();
    }
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_r);
        gtap_finalize();
        return 1;
    }
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(r.data(), d_r, r_count * sizeof(Tile), cudaMemcpyDeviceToHost);
    const bool ok = rectmul_host::verify(a, b, r, opts.x, opts.y, opts.z);
    std::printf("Execution time: %.3f ms\n", ms);

#ifdef GTAP_ENABLE_PROFILING
    gtap_export_profile({
        .output_directory = "./profile/rectmul_block_ychunk",
        .overwrite = true,
    });
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_r);
    err = gtap_finalize();
    return ok && err == cudaSuccess ? 0 : 1;
}
