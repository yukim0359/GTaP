// Port of CUDA Samples v13.4 cdpSimpleQuicksort onto GTaP thread mode.
// Partition, depth limit, cutoff predicate, and selection sort match
// cpp/3_CUDA_Features/cdpSimpleQuicksort/cdpSimpleQuicksort.cu.
// Child recursion is a GTaP task plus taskwait instead of a <<<1,1>>> CDP launch.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include "gtap_thread.cuh"

#define MAX_DEPTH      16
#define INSERTION_SORT 32

__device__ void selection_sort(int *data, int left, int right)
{
    for (int i = left; i <= right; ++i) {
        int min_val = data[i];
        int min_idx = i;
        for (int j = i + 1; j <= right; ++j) {
            int val_j = data[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__device__ void partition_range(int *data, int left, int right, int *nleft, int *nright)
{
    int *lptr = data + left;
    int *rptr = data + right;
    int pivot = data[(left + right) / 2];

    while (lptr <= rptr) {
        int lval = *lptr;
        int rval = *rptr;

        while (lval < pivot) {
            lptr++;
            lval = *lptr;
        }
        while (rval > pivot) {
            rptr--;
            rval = *rptr;
        }
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    *nright = static_cast<int>(rptr - data);
    *nleft = static_cast<int>(lptr - data);
}

#pragma gtap function
__device__ void quicksort(int *data, int left, int right, int depth)
{
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    int nleft = 0;
    int nright = 0;
    partition_range(data, left, right, &nleft, &nright);

    if (left < nright) {
        #pragma gtap task
        quicksort(data, left, nright, depth + 1);
    }
    if (nleft < right) {
        #pragma gtap task
        quicksort(data, nleft, right, depth + 1);
    }
    #pragma gtap taskwait
}

__global__ void exec_kernel(int *data, int n)
{
    #pragma gtap entry
    quicksort(data, 0, n - 1, 0);
}

static std::vector<int> load_array(const char* filename, size_t& n) {
    std::vector<int> data;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
        return data;
    }

    if (fread(&n, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read size from %s\n", filename);
        fclose(fp);
        return data;
    }

    data.resize(n);
    if (fread(data.data(), sizeof(int), n, fp) != n) {
        fprintf(stderr, "Error: Cannot read data from %s\n", filename);
        data.clear();
        fclose(fp);
        return data;
    }

    fclose(fp);
    return data;
}

int main(int argc, char** argv) {
    size_t N = 500000;
    std::vector<int> h;
    const char* reference_file = nullptr;
    if (argc >= 2) {
        const char* data_file = argv[1];
        h = load_array(data_file, N);
        if (h.empty()) return 1;
        printf("Loaded %zu elements from %s\n", N, data_file);
        if (argc >= 3) reference_file = argv[2];
    } else {
        std::mt19937 rng(12345);
        h.resize(N);
        for (size_t i = 0; i < N; ++i) h[i] = static_cast<int>(rng());
        printf("Generated %zu random elements (use %s <file> [reference] for custom data)\n", N, argv[0]);
    }

    std::vector<int> gold;
    if (reference_file) {
        size_t ref_n = 0;
        gold = load_array(reference_file, ref_n);
        if (gold.empty()) return 1;
        if (ref_n != N) {
            fprintf(stderr, "Error: Reference file size mismatch\n");
            return 1;
        }
        printf("Loaded %zu reference elements from %s\n", ref_n, reference_file);
    } else {
        gold = h;
        std::sort(gold.begin(), gold.end());
    }

    gtap_thread_config config{
        .grid_size = 1000,
        .block_size = 32,
        .max_tasks_per_warp = 80000,
    };
    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        return 1;
    }

    int* d_data = nullptr;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMemcpy(d_data, h.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, d_data, static_cast<int>(N));
    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        err = gtap_synchronize();
    }
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        gtap_finalize();
        return 1;
    }
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h.data(), d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    bool ok = (h == gold);
    printf("Quicksort(%zu) = %s\n", N, ok ? "Correct" : "Incorrect");
    printf("Execution time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    err = gtap_finalize();
    return ok && err == cudaSuccess ? 0 : 1;
}
