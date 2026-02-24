#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include "gtap_thread.cuh"

// Divide-and-conquer SpMV: 1 thread = 1 task
// range_spmv recursively splits the row range; leaf tasks compute one row.

#define RANGE_CUTOFF 16

// Device globals
__device__ int*    g_row_ptr;
__device__ int*    g_col_idx;
__device__ double* g_val;
__device__ double* g_x;
__device__ double* g_y;

static inline cudaError_t bind_device_arrays(
    int* d_row_ptr, int* d_col_idx, double* d_val,
    double* d_x, double* d_y) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_row_ptr, &d_row_ptr, sizeof(d_row_ptr)); if (st) return st;
    st = cudaMemcpyToSymbol(g_col_idx, &d_col_idx, sizeof(d_col_idx)); if (st) return st;
    st = cudaMemcpyToSymbol(g_val,     &d_val,     sizeof(d_val));     if (st) return st;
    st = cudaMemcpyToSymbol(g_x,       &d_x,       sizeof(d_x));       if (st) return st;
    st = cudaMemcpyToSymbol(g_y,       &d_y,       sizeof(d_y));       if (st) return st;
    return cudaSuccess;
}

// Compute y[row_idx] = sum_k A[row_idx, k] * x[k]
#pragma gtap function
__device__ void spmv_row(int row_idx) {
    double sum = 0.0;
    int row_begin = g_row_ptr[row_idx];
    int row_end   = g_row_ptr[row_idx + 1];
    for (int k = row_begin; k < row_end; k++) {
        sum += g_val[k] * g_x[g_col_idx[k]];
    }
    g_y[row_idx] = sum;
}

// Divide [begin, end) into halves; spawn spmv_row at leaves
#pragma gtap function
__device__ void range_spmv(int begin, int end) {
    if (end - begin <= RANGE_CUTOFF) {
        for (int i = begin; i < end; i++) {
            #pragma gtap task
            spmv_row(i);
        }
    } else {
        int mid = (begin + end) >> 1;
        #pragma gtap task
        range_spmv(begin, mid);
        #pragma gtap task
        range_spmv(mid, end);
    }
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    range_spmv(0, n);
}

// ---- Host utilities ----

static bool load_csr(const char* path, int& rows, int& cols, int& nnz,
                     std::vector<int>& row_ptr, std::vector<int>& col_idx,
                     std::vector<double>& val) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.read(reinterpret_cast<char*>(&rows), sizeof(int));
    f.read(reinterpret_cast<char*>(&cols), sizeof(int));
    f.read(reinterpret_cast<char*>(&nnz),  sizeof(int));
    if (!f) return false;
    row_ptr.resize(rows + 1);
    col_idx.resize(nnz);
    val.resize(nnz);
    f.read(reinterpret_cast<char*>(row_ptr.data()), (rows + 1) * sizeof(int));
    f.read(reinterpret_cast<char*>(col_idx.data()), nnz * sizeof(int));
    f.read(reinterpret_cast<char*>(val.data()),     nnz * sizeof(double));
    return f.good();
}

// Generate a random CSR matrix (fallback when no file is given)
static void gen_random_csr(int N, int avg_nnz, std::mt19937& rng,
                            int& rows, int& cols, int& nnz,
                            std::vector<int>& row_ptr, std::vector<int>& col_idx,
                            std::vector<double>& val) {
    rows = cols = N;
    std::uniform_int_distribution<int> col_dist(0, N - 1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
    std::uniform_int_distribution<int> cnt_dist(1, avg_nnz * 2 - 1);

    row_ptr.resize(N + 1);
    row_ptr[0] = 0;
    for (int i = 0; i < N; i++) {
        int cnt = cnt_dist(rng);
        std::vector<int> cols_in_row(cnt);
        for (int k = 0; k < cnt; k++) cols_in_row[k] = col_dist(rng);
        std::sort(cols_in_row.begin(), cols_in_row.end());
        cols_in_row.erase(std::unique(cols_in_row.begin(), cols_in_row.end()), cols_in_row.end());
        for (int c : cols_in_row) { col_idx.push_back(c); val.push_back(val_dist(rng)); }
        row_ptr[i + 1] = (int)col_idx.size();
    }
    nnz = (int)col_idx.size();
}

int main(int argc, char** argv) {
    int rows = 0, cols = 0, nnz = 0;
    std::vector<int>    row_ptr, col_idx;
    std::vector<double> val;

    if (argc >= 2) {
        if (!load_csr(argv[1], rows, cols, nnz, row_ptr, col_idx, val)) {
            fprintf(stderr, "Failed to load CSR file: %s\n", argv[1]);
            return 1;
        }
        printf("Loaded CSR: rows=%d cols=%d nnz=%d from %s\n", rows, cols, nnz, argv[1]);
    } else {
        std::mt19937 rng(42);
        gen_random_csr(1000000, 300, rng, rows, cols, nnz, row_ptr, col_idx, val);
        printf("Using synthetic CSR: rows=%d cols=%d nnz=%d\n", rows, cols, nnz);
    }

    std::vector<double> h_x(cols, 1.0);
    std::vector<double> h_y(rows, 0.0);

    int*    d_row_ptr = nullptr;
    int*    d_col_idx = nullptr;
    double* d_val     = nullptr;
    double* d_x       = nullptr;
    double* d_y       = nullptr;

    cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx,  nnz       * sizeof(int));
    cudaMalloc(&d_val,      nnz       * sizeof(double));
    cudaMalloc(&d_x,        cols      * sizeof(double));
    cudaMalloc(&d_y,        rows      * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx.data(),  nnz       * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,     val.data(),      nnz       * sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,       h_x.data(),     cols      * sizeof(double),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,       h_y.data(),     rows      * sizeof(double),  cudaMemcpyHostToDevice);

    cudaError_t st = gtap_initialize();
    if (st != cudaSuccess) {
        fprintf(stderr, "gtap_initialize failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    st = bind_device_arrays(d_row_ptr, d_col_idx, d_val, d_x, d_y);
    if (st != cudaSuccess) {
        fprintf(stderr, "bind_device_arrays failed: %s\n", cudaGetErrorString(st));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    exec_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(rows);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Execution time: %.3f ms\n", ms);

    // CPU reference SpMV for validation
    std::vector<double> h_y_cpu(rows, 0.0);
    for (int i = 0; i < rows; i++) {
        double s = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++)
            s += val[k] * h_x[col_idx[k]];
        h_y_cpu[i] = s;
    }

    double max_err = 0.0, sum_y = 0.0;
    for (int i = 0; i < rows; i++) {
        sum_y += h_y[i];
        double err = fabs(h_y[i] - h_y_cpu[i]);
        if (err > max_err) max_err = err;
    }
    printf("Sum of y: %.6f\n", sum_y);
    double max_rel_err = 0.0;
    for (int i = 0; i < rows; i++) {
        double ref = fabs(h_y_cpu[i]);
        double rel = (ref > 0.0) ? fabs(h_y[i] - h_y_cpu[i]) / ref : fabs(h_y[i] - h_y_cpu[i]);
        if (rel > max_rel_err) max_rel_err = rel;
    }
    printf("Max error (vs CPU): %.2e (abs)  %.2e (rel)  [%s]\n",
           max_err, max_rel_err, max_rel_err < 1e-6 ? "PASSED" : "FAILED");

#ifdef PROFILE
    gtap_visualize_profile("spmv_thread");
#endif

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
