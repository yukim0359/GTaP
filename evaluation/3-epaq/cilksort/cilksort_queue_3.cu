#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
// #define DEBUG
#include "gtap_thread.cuh"

// #define TASK_SPAWN_CUTOFF_SORT 64
// #define TASK_SPAWN_CUTOFF_MERGE 256

__device__ int* g_data;
__device__ int* g_buf;

// Sequential merge
__device__ __forceinline__ void merge_device(int* a, int a_len, int* b, int b_len, int* dst) {
    int i = 0, j = 0, ti = 0;
    while (i < a_len && j < b_len) {
        int a_val = load_L2(&a[i]);
        int b_val = load_L2(&b[j]);
        if (a_val < b_val) {
            dst[ti++] = a_val;
            i++;
        } else {
            dst[ti++] = b_val;
            j++;
        }
    }
    while (i < a_len) {
        dst[ti++] = load_L2(&a[i++]);
    }
    while (j < b_len) {
        dst[ti++] = load_L2(&b[j++]);
    }
}

// Binary search: return the largest index s.t. x[index] <= val
__device__ __forceinline__ int binary_search_device(int* x, int len, int val) {
    int low = -1;
    int high = len;
    while (low + 1 < high) {
        int mid = (low + high) / 2;
        int mid_val = load_L2(&x[mid]);
        if (mid_val <= val) low = mid;
        else high = mid;
    }
    return low;
}

__device__ __forceinline__ void sort_small_device(int* arr, int n) {
    int buf[TASK_SPAWN_CUTOFF_SORT];

    #pragma unroll
    for (int i = 0; i < TASK_SPAWN_CUTOFF_SORT; ++i) {
        if (i < n) {
            buf[i] = load_L2(&arr[i]);
        } else {
            buf[i] = INT_MAX;
        }
    }

    for (int k = 2; k <= TASK_SPAWN_CUTOFF_SORT; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int i = 0; i < TASK_SPAWN_CUTOFF_SORT; ++i) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    int a = buf[i];
                    int b = buf[ixj];
                    bool cond = up ? (a > b) : (a < b);
                    if (cond) {
                        buf[i]   = b;
                        buf[ixj] = a;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < n; ++i) {
        arr[i] = buf[i];
    }
}

// Parallel merge task
#pragma gtap function
__device__ void cilkmerge(int* a, int a_len, int* b, int b_len, int* dst) {
    // Ensure a_len >= b_len
    if (a_len < b_len) {
        int* tmp = a; a = b; b = tmp;
        int tmp2 = a_len; a_len = b_len; b_len = tmp2;
    }

    // Base case: empty arrays
    if (b_len == 0) {
        // Copy a to dst if a_len > 0
        if (a_len > 0) {
            for (int i = 0; i < a_len; i++) {
                dst[i] = load_L2(&a[i]);
            }
            // printf("cilkmerge: a_len = %d\n", a_len);
        }
        __threadfence();
        return;
    }

    // Sequential merge for small arrays
    if (a_len + b_len < TASK_SPAWN_CUTOFF_MERGE) {
        merge_device(a, a_len, b, b_len, dst);
        return;
    }

    // Parallel merge: split and recurse
    int a_split = (a_len + 1) / 2;
    int b_split = binary_search_device(b, b_len, load_L2(&a[a_split - 1])) + 1;

    #pragma gtap task queue(a_split + b_split < TASK_SPAWN_CUTOFF_MERGE ? 2 : 0)
    cilkmerge(a, a_split, b, b_split, dst);
    #pragma gtap task queue((a_len - a_split) + (b_len - b_split) < TASK_SPAWN_CUTOFF_MERGE ? 2 : 0)
    cilkmerge(a + a_split, a_len - a_split, b + b_split, b_len - b_split, dst + a_split + b_split);
    #pragma gtap taskwait
    return;
}

// Sort task
#pragma gtap function
__device__ void cilksort(int* arr, int n, int* tmp) {
    if (n < TASK_SPAWN_CUTOFF_SORT) {
        sort_small_device(arr, n);
        return;
    }

    // Cilksort: divide into 4 parts
    int len12 = n / 2;
    int len1 = len12 / 2;
    int len2 = len12 - len1;
    int len34 = n - len12;
    int len3 = len34 / 2;
    int len4 = len34 - len3;

    // Spawn 4 sort tasks
    #pragma gtap task queue(len1 < TASK_SPAWN_CUTOFF_SORT ? 1 : 0)
    cilksort(arr, len1, tmp);
    #pragma gtap task queue(len2 < TASK_SPAWN_CUTOFF_SORT ? 1 : 0)
    cilksort(arr + len1, len2, tmp + len1);
    #pragma gtap task queue(len3 < TASK_SPAWN_CUTOFF_SORT ? 1 : 0)
    cilksort(arr + len12, len3, tmp + len12);
    #pragma gtap task queue(len4 < TASK_SPAWN_CUTOFF_SORT ? 1 : 0)
    cilksort(arr + len12 + len3, len4, tmp + len12 + len3);
    #pragma gtap taskwait

    #pragma gtap task queue(len1 + len2 < TASK_SPAWN_CUTOFF_MERGE ? 2 : 0)
    cilkmerge(arr, len1, arr + len1, len2, tmp);
    #pragma gtap task queue(len3 + len4 < TASK_SPAWN_CUTOFF_MERGE ? 2 : 0)
    cilkmerge(arr + len12, len3, arr + len12 + len3, len4, tmp + len12);
    #pragma gtap taskwait

    #pragma gtap task queue(len12 + len34 < TASK_SPAWN_CUTOFF_MERGE ? 2 : 0)
    cilkmerge(tmp, len12, tmp + len12, len34, arr);
    #pragma gtap taskwait

    return;
}

__global__ void my_kernel(int n) {
    #pragma gtap entry
    cilksort(g_data, n, g_buf);
}

// Helpers to bind device pointers
static inline cudaError_t bind_device_arrays(int* d_data, int* d_buf) {
    cudaError_t st;
    st = cudaMemcpyToSymbol(g_data, &d_data, sizeof(d_data)); if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_buf,  &d_buf,  sizeof(d_buf));  if (st != cudaSuccess) return st;
    return cudaSuccess;
}

// Load array from binary file
std::vector<int> load_array(const char* filename, size_t& n) {
    std::vector<int> data;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for reading\n", filename);
        return data;
    }
    
    // Read size
    if (fread(&n, sizeof(size_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read size from %s\n", filename);
        fclose(fp);
        return data;
    }
    
    // Allocate and read data
    data.resize(n);
    if (fread(data.data(), sizeof(int), n, fp) != n) {
        fprintf(stderr, "Error: Cannot read data from %s\n", filename);
        data.clear();
        fclose(fp);
        return data;
    }
    
    fclose(fp);
    printf("Loaded %zu elements from %s\n", n, filename);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <data_file> [reference_file]\n", argv[0]);
        printf("  data_file: Binary file containing array data (generated by gen_vector)\n");
        printf("  reference_file: Optional reference file for verification\n");
        return 1;
    }
    
    const char* data_file = argv[1];
    const char* reference_file = (argc > 2) ? argv[2] : nullptr;

    cudaError_t err = gtap_initialize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Load data from file
    size_t N;
    std::vector<int> h = load_array(data_file, N);
    if (h.empty()) return 1;
    
    // Create reference for verification
    std::vector<int> gold;
    if (reference_file) {
        size_t ref_n;
        gold = load_array(reference_file, ref_n);
        if (gold.empty() || ref_n != N) {
            fprintf(stderr, "Error: Reference file size mismatch\n");
            return 1;
        }
    } else {
        gold = h;
        std::sort(gold.begin(), gold.end());
    }

    // Device buffers
    int* d_data = nullptr; int* d_buf = nullptr;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMalloc(&d_buf,  sizeof(int) * N);
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
    // Copy back and verify
    cudaMemcpy(h.data(), d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    std::sort(gold.begin(), gold.end());
    bool ok = std::is_sorted(h.begin(), h.end()) && (h == gold);

    // Match mergesort.cu output style
    printf("Cilksort(%zu) = %s\n", N, ok ? "Correct" : "Incorrect");
    printf("Execution time: %.3f ms\n", ms);

    cudaFree(d_data); cudaFree(d_buf);
    return ok ? 0 : 1;
}
