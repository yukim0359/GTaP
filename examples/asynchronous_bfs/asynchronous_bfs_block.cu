#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <queue>
#include <algorithm>
#include <time.h>
#include "gtap_block.cuh"

// Device-side graph state
__device__ int* g_row_offsets;   // size: num_vertices + 1
__device__ int* g_col_indices;   // size: num_edges
__device__ int* g_depth;         // size: num_vertices; INF indicates unvisited
__device__ int  g_num_vertices;  // number of vertices

// All threads in the block iterate over the adjacency list of v with stride.
// Each thread that discovers an unvisited neighbor spawns a child task.
#pragma gtap function
__device__ void bfs(int v) {
    int dv = g_depth[v];
    int row_start = g_row_offsets[v];
    int row_end   = g_row_offsets[v + 1];

    for (int e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
        int u = g_col_indices[e];
        int old = atomicMin(&g_depth[u], dv + 1);
        if (old > dv + 1) {
            #pragma gtap task
            bfs(u);
        }
    }
}

__global__ void exec_kernel(int source) {
    g_depth[source] = 0;
    #pragma gtap entry
    bfs(source);
}

// ---- Host utilities ----

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

int main(int argc, char** argv) {
    std::string csr_path;
    int N = 1000;
    int source = 0;
    if (argc >= 2) csr_path = argv[1];
    if (argc >= 3) source = atoi(argv[2]);

    cudaError_t err = gtap_initialize();
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

    int *d_row = nullptr, *d_col = nullptr, *d_depth = nullptr;
    cudaMalloc(&d_row,   sizeof(int) * (N + 1));
    cudaMalloc(&d_col,   sizeof(int) * M);
    cudaMalloc(&d_depth, sizeof(int) * N);
    cudaMemcpy(d_row, h_row, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, sizeof(int) * M,       cudaMemcpyHostToDevice);
    cudaMemset(d_depth, 0x3f, sizeof(int) * N);

    cudaMemcpyToSymbol(g_row_offsets,  &d_row,   sizeof(d_row));
    cudaMemcpyToSymbol(g_col_indices,  &d_col,   sizeof(d_col));
    cudaMemcpyToSymbol(g_depth,        &d_depth, sizeof(d_depth));
    cudaMemcpyToSymbol(g_num_vertices, &N,       sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    exec_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(source);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    int* h_depth = (int*)malloc(sizeof(int) * N);
    cudaMemcpy(h_depth, d_depth, sizeof(int) * N, cudaMemcpyDeviceToHost);
    printf("BFS done. depth[source]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d, depth[%d]=%d\n",
           h_depth[source], 1, h_depth[1], 2, h_depth[2], 3, h_depth[3], 4, h_depth[4], 5, h_depth[5]);
    printf("Execution time: %.3f ms\n", ms);

    // Validation: Compare with CPU reference BFS
    printf("\n=== BFS Validation ===\n");
    int* h_depth_cpu = (int*)malloc(sizeof(int) * N);
    const int INF_CPU = 0x3f3f3f3f;
    for (int i = 0; i < N; i++) h_depth_cpu[i] = INF_CPU;

    std::queue<int> q;
    h_depth_cpu[source] = 0;
    q.push(source);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (!q.empty()) {
        int v = q.front(); q.pop();
        int depth_v = h_depth_cpu[v];
        for (int e = h_row[v]; e < h_row[v + 1]; e++) {
            int u = h_col[e];
            if (depth_v + 1 < h_depth_cpu[u]) {
                h_depth_cpu[u] = depth_v + 1;
                q.push(u);
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("BFS CPU time: %.3f ms\n",
           (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1000000.0);

    int error_count = 0;
    for (int i = 0; i < N; i++) {
        if (h_depth_cpu[i] != h_depth[i]) {
            error_count++;
            if (error_count <= 20)
                printf("ERROR: depth[%d] mismatch: CPU=%d, GPU=%d\n", i, h_depth_cpu[i], h_depth[i]);
        }
    }
    printf("Validation results:\n");
    printf("  Total vertices: %d\n", N);
    printf("  Errors found: %d\n", error_count);
    if (error_count == 0)
        printf("  ✓ BFS results are CORRECT!\n");
    else
        printf("  ✗ BFS results have ERRORS!\n");

    printf("\nFirst 20 depths comparison:\n");
    printf("CPU: "); for (int i = 0; i < 20 && i < N; i++) printf("%d ", h_depth_cpu[i]); printf("\n");
    printf("GPU: "); for (int i = 0; i < 20 && i < N; i++) printf("%d ", h_depth[i]);     printf("\n");

    free(h_depth_cpu);
    free(h_row); free(h_col); free(h_depth);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth);
    gtap_finalize();
    return 0;
}
