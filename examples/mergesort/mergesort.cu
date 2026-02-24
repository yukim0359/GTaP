#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include "gtap_thread.cuh"

__device__ int* g_data;
__device__ int* g_buf;

#define TASK_SPAWN_CUTOFF 128

__device__ __forceinline__ void merge_device(int* a, int* buf, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    while (i <= mid && j <= right) {
        if (a[i] <= a[j])  buf[k++] = a[i++];
        else               buf[k++] = a[j++];
    }
    while (i <= mid)   buf[k++] = a[i++];
    while (j <= right) buf[k++] = a[j++];
    for (int t = left; t <= right; ++t) a[t] = buf[t];
}

__device__ __forceinline__ void mergesort_rec_device(int* a, int left, int right) {
    if (left >= right) return;
    int mid = left + ((right - left) >> 1);
    mergesort_rec_device(a, left, mid);
    mergesort_rec_device(a, mid + 1, right);
    merge_device(a, g_buf, left, mid, right);
}

#pragma gtap function
__device__ void mergesort(int left, int right) {
    int n = right - left + 1;
    if (left >= right) return;
    int mid = left + ((right - left) >> 1);
    if (n > TASK_SPAWN_CUTOFF) {
        #pragma gtap task
        mergesort(left, mid);
        #pragma gtap task
        mergesort(mid + 1, right);
        #pragma gtap taskwait
    } else {
        mergesort_rec_device(g_data, left, right);
        return;
    }
    merge_device(g_data, g_buf, left, mid, right);
}

__global__ void my_kernel(int n) {
    #pragma gtap entry
    mergesort(0, n - 1);
}

static inline cudaError_t bind_device_arrays(int* d_data, int* d_buf) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_data, &d_data, sizeof(d_data)); if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_buf,  &d_buf,  sizeof(d_buf));  if (st != cudaSuccess) return st;
    return cudaSuccess;
}

int main(int argc, char** argv) {
    size_t N = 500000;
    std::vector<int> h;
    if (argc >= 2) {
        const char* data_file = argv[1];
        FILE* fp = fopen(data_file, "rb");
        if (!fp) {
            fprintf(stderr, "Cannot open %s\n", data_file);
            return 1;
        }
        if (fread(&N, sizeof(size_t), 1, fp) != 1 || N == 0) {
            fclose(fp);
            return 1;
        }
        h.resize(N);
        if (fread(h.data(), sizeof(int), N, fp) != N) {
            h.clear();
            fclose(fp);
            return 1;
        }
        fclose(fp);
        printf("Loaded %zu elements from %s\n", N, data_file);
    } else {
        std::mt19937 rng(12345);
        h.resize(N);
        for (size_t i = 0; i < N; ++i) h[i] = static_cast<int>(rng());
        printf("Generated %zu random elements (use %s <file> for custom data)\n", N, argv[0]);
    }

    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<int> gold = h;
    std::sort(gold.begin(), gold.end());

    int* d_data = nullptr;
    int* d_buf = nullptr;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMalloc(&d_buf, sizeof(int) * N);
    cudaMemcpy(d_data, h.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
    bind_device_arrays(d_data, d_buf);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    my_kernel<<<GTAP_GRID_SIZE, GTAP_BLOCK_SIZE>>>(static_cast<int>(N));
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h.data(), d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    bool ok = (h == gold);
    printf("Mergesort(%zu) = %s\n", N, ok ? "OK" : "FAIL");
    printf("Execution time: %.3f ms\n", ms);

    cudaFree(d_data);
    cudaFree(d_buf);
    return ok ? 0 : 1;
}
