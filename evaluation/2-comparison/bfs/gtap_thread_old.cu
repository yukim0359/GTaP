#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <queue>
#include <algorithm>

// #define PROFILE
#define GTAP_MAX_TASK_DATA_SIZE 8
#include "gtap_thread.cuh"

// Device-side graph state
__device__ int* g_row_offsets;   // size: num_vertices + 1
__device__ int* g_col_indices;   // size: num_edges
__device__ int* g_depth;         // size: num_vertices; INF indicates unvisited
__device__ int  g_num_vertices;  // number of vertices

#pragma gtap function worker_size(thread)
__device__ void bfs(int v) {
    int dv = load_L2(&g_depth[v]);
    int row_start = g_row_offsets[v];
    int row_end   = g_row_offsets[v + 1];

    for (int e = row_start; e < row_end; ++e) {
        int u = g_col_indices[e];
        int old = atomicMin(&g_depth[u], dv + 1);
        if (old > dv + 1) {
            #pragma gtap task
            bfs(u);
        }
    }
}

__global__ void my_kernel(int source) {
    g_depth[source] = 0;
    #pragma gtap entry
    bfs(source);
}


// Simple demo host: build a chain graph 0->1->2->...->(N-1)
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
    int N = 1000; // default chain graph vertices
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
    cudaMalloc(&d_row, sizeof(int) * (N + 1));
    cudaMalloc(&d_col, sizeof(int) * M);
    cudaMalloc(&d_depth, sizeof(int) * N);
    cudaMemcpy(d_row, h_row, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, h_col, sizeof(int) * M, cudaMemcpyHostToDevice);
    cudaMemset(d_depth, 0x3f, sizeof(int) * N);

    // Bind device pointers to device symbols
    cudaMemcpyToSymbol(g_row_offsets, &d_row, sizeof(d_row));
    cudaMemcpyToSymbol(g_col_indices, &d_col, sizeof(d_col));
    cudaMemcpyToSymbol(g_depth, &d_depth, sizeof(d_depth));
    cudaMemcpyToSymbol(g_num_vertices, &N, sizeof(int));

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(source);
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

    // Validation: Compare with CPU reference implementation
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

    // Compare results
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
    
    if (error_count == 0) {
        printf("  ✓ BFS results are CORRECT!\n");
    } else {
        printf("  ✗ BFS results have ERRORS!\n");
    }
    
    // Print first 20 depths for comparison
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
    gtap_visualize_profile("bfs");
    #endif

    free(h_row); free(h_col); free(h_depth);
    cudaFree(d_row); cudaFree(d_col); cudaFree(d_depth);
    return 0;

}
