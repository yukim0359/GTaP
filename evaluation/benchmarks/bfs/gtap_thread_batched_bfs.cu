#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <queue>
#include <algorithm>

// #define PROFILE
#include "gtap_thread.cuh"

#ifndef GTAP_BENCH_GRID_SIZE
#define GTAP_BENCH_GRID_SIZE 4000
#endif
#ifndef GTAP_BENCH_BLOCK_SIZE
#define GTAP_BENCH_BLOCK_SIZE 32
#endif
#ifndef GTAP_BENCH_MAX_TASKS_PER_WARP
#define GTAP_BENCH_MAX_TASKS_PER_WARP 100000
#endif
#ifndef GTAP_BENCH_NUM_QUEUES
#define GTAP_BENCH_NUM_QUEUES 1
#endif

__device__ int* g_row_offsets;
__device__ int* g_col_indices;
__device__ int* g_depth;
__device__ int  g_num_vertices;

__device__ int* g_task_vertices;
__device__ int  g_task_tail;
__device__ int  g_task_capacity;
__device__ int  g_task_overflow;
__device__ unsigned long long g_discoveries;
__device__ unsigned long long g_batches_spawned;

#ifndef BFS_SPLIT_THRESHOLD
#define BFS_SPLIT_THRESHOLD 64
#endif

#ifndef BFS_BATCH_SIZE
#define BFS_BATCH_SIZE 16
#endif

#define BFS_TASK_BATCH 0
#define BFS_TASK_EDGES 1

#pragma gtap function
__device__ void bfs_task(int kind, int a, int b, int c) {
    int local_vertices[BFS_BATCH_SIZE];
    int local_count = 0;

    int item_count = (kind == BFS_TASK_BATCH) ? b : 1;
    for (int i = 0; i < item_count; ++i) {
        int v = (kind == BFS_TASK_BATCH) ? g_task_vertices[a + i] : a;
        int dv = load_L2(&g_depth[v]);
        int row_start = (kind == BFS_TASK_BATCH) ? g_row_offsets[v] : b;
        int row_end = (kind == BFS_TASK_BATCH) ? g_row_offsets[v + 1] : c;
        int deg = row_end - row_start;

        if (kind == BFS_TASK_BATCH && deg > BFS_SPLIT_THRESHOLD) {
            for (int s = row_start; s < row_end; s += BFS_SPLIT_THRESHOLD) {
                int e = s + BFS_SPLIT_THRESHOLD;
                if (e > row_end) e = row_end;

                atomicAdd(&g_batches_spawned, 1ULL);
                #pragma gtap task
                bfs_task(BFS_TASK_EDGES, v, s, e);
            }
            continue;
        }

        for (int e = row_start; e < row_end; ++e) {
            int u = g_col_indices[e];
            int old = atomicMin(&g_depth[u], dv + 1);
            if (old > dv + 1) {
                atomicAdd(&g_discoveries, 1ULL);
                local_vertices[local_count++] = u;

                if (local_count == BFS_BATCH_SIZE) {
                    int out = atomicAdd(&g_task_tail, BFS_BATCH_SIZE);
                    if (out + BFS_BATCH_SIZE <= g_task_capacity) {
                        for (int j = 0; j < BFS_BATCH_SIZE; ++j) {
                            g_task_vertices[out + j] = local_vertices[j];
                        }
                        atomicAdd(&g_batches_spawned, 1ULL);
                        #pragma gtap task
                        bfs_task(BFS_TASK_BATCH, out, BFS_BATCH_SIZE, 0);
                    } else {
                        g_task_overflow = 1;
                    }
                    local_count = 0;
                }
            }
        }
    }

    if (local_count > 0) {
        int out = atomicAdd(&g_task_tail, local_count);
        if (out + local_count <= g_task_capacity) {
            for (int i = 0; i < local_count; ++i) {
                g_task_vertices[out + i] = local_vertices[i];
            }
            atomicAdd(&g_batches_spawned, 1ULL);
            #pragma gtap task
            bfs_task(BFS_TASK_BATCH, out, local_count, 0);
        } else {
            g_task_overflow = 1;
        }
    }
}

__global__ void my_kernel(int source) {
    g_depth[source] = 0;
    g_task_vertices[0] = source;
    g_task_tail = 1;
    g_task_overflow = 0;
    g_discoveries = 1;
    g_batches_spawned = 1;

    #pragma gtap entry
    bfs_task(BFS_TASK_BATCH, 0, 1, 0);
}

static void build_chain_graph(int N, int** h_row, int** h_col, int* M_out) {
    int* row = (int*)malloc(sizeof(int) * (N + 1));
    int* col = (int*)malloc(sizeof(int) * (N - 1));
    for (int i = 0; i <= N; ++i) row[i] = i < N ? i : (N - 1);
    for (int i = 0; i < N - 1; ++i) col[i] = i + 1;
    *h_row = row;
    *h_col = col;
    *M_out = N - 1;
}

static bool read_csr_binary(const char* path, int** h_row, int** h_col, int* N_out, int* M_out) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;
    int N = 0, M = 0;
    fin.read(reinterpret_cast<char*>(&N), sizeof(int));
    fin.read(reinterpret_cast<char*>(&M), sizeof(int));
    if (!fin) return false;
    int* row = (int*)malloc(sizeof(int) * (N + 1));
    int* col = (int*)malloc(sizeof(int) * M);
    if (row == nullptr || col == nullptr) return false;
    fin.read(reinterpret_cast<char*>(row), sizeof(int) * (N + 1));
    fin.read(reinterpret_cast<char*>(col), sizeof(int) * M);
    if (!fin) { free(row); free(col); return false; }
    *h_row = row; *h_col = col; *N_out = N; *M_out = M;
    return true;
}

int main(int argc, char **argv) {
    std::string csr_path;
    int N = 1000;
    int source = 0;
    int requested_capacity = 0;
    if (argc >= 2) csr_path = argv[1];
    if (argc >= 3) source = atoi(argv[2]);
    if (argc >= 4) requested_capacity = atoi(argv[3]);

    gtap_thread_config config{
        .grid_size = GTAP_BENCH_GRID_SIZE,
        .block_size = GTAP_BENCH_BLOCK_SIZE,
        .max_tasks_per_warp = GTAP_BENCH_MAX_TASKS_PER_WARP,
        .num_queues = GTAP_BENCH_NUM_QUEUES,
    };
    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int *h_row = nullptr, *h_col = nullptr; int M = 0;
    if (!csr_path.empty()) {
        if (!read_csr_binary(csr_path.c_str(), &h_row, &h_col, &N, &M)) {
            fprintf(stderr, "Failed to read CSR file: %s\n", csr_path.c_str());
            return 1;
        }
        fprintf(stdout, "Loaded CSR: N=%d, M=%d from %s\n", N, M, csr_path.c_str());
        fflush(stdout);
    } else {
        build_chain_graph(N, &h_row, &h_col, &M);
        fprintf(stdout, "Using synthetic chain graph: N=%d, M=%d\n", N, M);
    }

    if (source < 0 || source >= N) {
        fprintf(stderr, "Invalid source vertex: %d (N=%d)\n", source, N);
        return 1;
    }

    long long default_capacity = (long long)N + (long long)M + 1;
    int task_capacity = requested_capacity > 0 ? requested_capacity : (int)default_capacity;
    if (default_capacity > 2147483647LL && requested_capacity <= 0) {
        fprintf(stderr, "Default task capacity exceeds int range: %lld\n", default_capacity);
        return 1;
    }

    int *d_row = nullptr, *d_col = nullptr, *d_depth = nullptr, *d_task_vertices = nullptr;
    cudaMalloc(&d_row, sizeof(int) * (N + 1));
    cudaMalloc(&d_col, sizeof(int) * M);
    cudaMalloc(&d_depth, sizeof(int) * N);
    cudaMalloc(&d_task_vertices, sizeof(int) * (size_t)task_capacity);
    cudaMemcpy(d_row, h_row, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, sizeof(int) * M, cudaMemcpyHostToDevice);
    cudaMemset(d_depth, 0x3f, sizeof(int) * N);

    cudaMemcpyToSymbol(g_row_offsets, &d_row, sizeof(d_row));
    cudaMemcpyToSymbol(g_col_indices, &d_col, sizeof(d_col));
    cudaMemcpyToSymbol(g_depth, &d_depth, sizeof(d_depth));
    cudaMemcpyToSymbol(g_num_vertices, &N, sizeof(int));
    cudaMemcpyToSymbol(g_task_vertices, &d_task_vertices, sizeof(d_task_vertices));
    cudaMemcpyToSymbol(g_task_capacity, &task_capacity, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(my_kernel, source);
    if (err != cudaSuccess) {
        fprintf(stderr, "gtap_launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    int h_task_tail = 0, h_task_overflow = 0;
    unsigned long long h_discoveries = 0, h_batches_spawned = 0;
    cudaMemcpyFromSymbol(&h_task_tail, g_task_tail, sizeof(int));
    cudaMemcpyFromSymbol(&h_task_overflow, g_task_overflow, sizeof(int));
    cudaMemcpyFromSymbol(&h_discoveries, g_discoveries, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_batches_spawned, g_batches_spawned, sizeof(unsigned long long));

    int* h_depth = (int*)malloc(sizeof(int) * N);
    cudaMemcpy(h_depth, d_depth, sizeof(int) * N, cudaMemcpyDeviceToHost);
    printf("BFS done. depth[source]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d\n",
           h_depth[source], 1, h_depth[1], 2, h_depth[2], 3, h_depth[3], 4, h_depth[4], 5, h_depth[5]);
    printf("Execution time: %.3f ms\n", ms);
    printf("Batched task stats: batch_size=%d, task_tail=%d, capacity=%d, overflow=%d, discoveries=%llu, batches_spawned=%llu\n",
           BFS_BATCH_SIZE, h_task_tail, task_capacity, h_task_overflow, h_discoveries, h_batches_spawned);

    printf("\n=== BFS Validation ===\n");
    int* h_depth_cpu = (int*)malloc(sizeof(int) * N);
    const int INF_CPU = 0x3f3f3f3f;
    for (int i = 0; i < N; i++) {
        h_depth_cpu[i] = INF_CPU;
    }

    std::queue<int> q;
    h_depth_cpu[source] = 0;
    q.push(source);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        int depth_v = h_depth_cpu[v];

        int row_start = h_row[v];
        int row_end = h_row[v + 1];

        for (int e = row_start; e < row_end; e++) {
            int u = h_col[e];
            int new_depth = depth_v + 1;

            if (new_depth < h_depth_cpu[u]) {
                h_depth_cpu[u] = new_depth;
                q.push(u);
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("BFS CPU time: %.3f ms\n", (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec)/1000000.0);

    int error_count = 0;
    int total_error = 0;
    int max_errors_to_print = 20;

    for (int i = 0; i < N; i++) {
        int diff = abs(h_depth_cpu[i] - h_depth[i]);
        total_error += diff;

        if (h_depth_cpu[i] != h_depth[i]) {
            error_count++;
            if (error_count <= max_errors_to_print) {
                printf("ERROR: depth[%d] mismatch: CPU=%d, GPU=%d\n",
                       i, h_depth_cpu[i], h_depth[i]);
            }
        }
    }

    printf("Validation results:\n");
    printf("  Total vertices: %d\n", N);
    printf("  Errors found: %d\n", error_count);
    printf("  Total error sum: %d\n", total_error);

    if (h_task_overflow != 0) {
        printf("  BFS task buffer overflowed; increase the optional task_capacity argument.\n");
    }
    if (error_count == 0 && h_task_overflow == 0) {
        printf("  BFS results are CORRECT!\n");
    } else {
        printf("  BFS results have ERRORS!\n");
    }

    printf("\nFirst 20 depths comparison:\n");
    printf("CPU: ");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", h_depth_cpu[i]);
    }
    printf("\nGPU: ");
    for (int i = 0; i < 20 && i < N; i++) {
        printf("%d ", h_depth[i]);
    }
    printf("\n");

    free(h_depth_cpu);

    #ifdef PROFILE
    gtap_visualize_profile("bfs_batched");
    #endif

    free(h_row); free(h_col); free(h_depth);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth); cudaFree(d_task_vertices);
    return (error_count == 0 && h_task_overflow == 0) ? 0 : 2;
}
