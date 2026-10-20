// cdpAdvancedQuicksort, expressed in GTaP block mode.
//
// The sample launches qsort_warp<<<ceil(len/512), 512>>>. GTaP has no
// dynamically sized child grid: one task is one block, and the block size is
// fixed at gtap_initialize. This port uses 1024, the sample's bitonic length.
// A partition of length L is at most 1024 block tasks. Each task walks several
// chunks of 1024 elements, then the parent joins them with taskwait.
// Each task owns one element per thread and splits with a warp ballot and
// atomics into a ping-pong buffer. Leaves up to 1024 elements are a one-block
// bitonic sort. A partition at depth 16 launches that large bitonic sort for
// any child still longer than 1024.

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <random>
#include <algorithm>

#include "gtap_block.cuh"
#include <cuda_runtime.h>

static constexpr unsigned BLOCK = 1024;
static constexpr unsigned BLOCK_SHIFT = 10; // log2(BLOCK). Chunk b owns elements [b*BLOCK, (b+1)*BLOCK).
static_assert((1u << BLOCK_SHIFT) == BLOCK, "BLOCK must be a power of two");
// One task walks many chunks. Spawning one task per chunk puts ceil(n/BLOCK)
// live ids on the parent block, and the runtime keeps 100 ids in reserve.
static constexpr int kPartitionFanout = 1024;
static constexpr unsigned MAX_DEPTH = 16;
static constexpr int PART_CAP = 1 << 23;

struct Part {
    unsigned lt;
    unsigned gt;
    unsigned sorted;
};

__device__ int d_part_cursor;
__device__ int d_part_overflow;

__device__ __forceinline__ unsigned floor_log2(unsigned word) {
    unsigned ret;
    asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
    return ret;
}

__device__ void bitonic_small(int *in, int *out, unsigned offset,
                              unsigned len) {
    __shared__ int buf[BLOCK];
    buf[threadIdx.x] = threadIdx.x < len ? in[offset + threadIdx.x] : INT_MAX;
    __syncthreads();

    for (unsigned k = 2; k <= BLOCK; k *= 2) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            unsigned swap_idx = threadIdx.x ^ j;
            int my_elem = buf[threadIdx.x];
            int swap_elem = buf[swap_idx];
            __syncthreads();

            unsigned ascend = k * (swap_idx < threadIdx.x);
            unsigned descend = k * (swap_idx > threadIdx.x);
            bool swap = false;
            if ((threadIdx.x & k) == ascend && my_elem > swap_elem)
                swap = true;
            if ((threadIdx.x & k) == descend && my_elem < swap_elem)
                swap = true;
            if (swap)
                buf[swap_idx] = my_elem;
            __syncthreads();
        }
    }

    if (threadIdx.x < len)
        out[offset + threadIdx.x] = buf[threadIdx.x];
}

__device__ void big_bitonic(int *data, int *dest, int *pad,
                            unsigned offset, unsigned len) {
    unsigned len2 = 1u << (floor_log2(len - 1u) + 1u);

    for (unsigned index = len + threadIdx.x; index < len2; index += BLOCK)
        pad[offset + index - len] = INT_MAX;
    __syncthreads();

    for (unsigned k = 2; k <= len2; k *= 2) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            for (unsigned base = 0; base < len2; base += BLOCK) {
                unsigned index = threadIdx.x + base;
                unsigned swap_idx = index ^ j;
                if (index < len2 && swap_idx > index && swap_idx < len2) {
                    int my_elem =
                        index < len ? data[offset + index] : pad[offset + index - len];
                    int swap_elem = swap_idx < len ? data[offset + swap_idx]
                                                   : pad[offset + swap_idx - len];
                    bool do_swap = false;
                    if ((index & k) == 0 && my_elem > swap_elem)
                        do_swap = true;
                    if ((index & k) == k && my_elem < swap_elem)
                        do_swap = true;
                    if (do_swap) {
                        if (swap_idx < len)
                            data[offset + swap_idx] = my_elem;
                        else
                            pad[offset + swap_idx - len] = my_elem;
                        if (index < len)
                            data[offset + index] = swap_elem;
                        else
                            pad[offset + index - len] = swap_elem;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (dest != data) {
        for (unsigned index = threadIdx.x; index < len; index += BLOCK)
            dest[offset + index] = data[offset + index];
    }
}

#pragma gtap function
__device__ void partition_block(int *in, int *out, unsigned offset,
                                unsigned len, Part *parts, int slot, int chunk,
                                int task_count, int nblocks) {
    int pivot = in[offset + len / 2];
    for (int b = chunk; b < nblocks; b += task_count) {
        unsigned thread_id =
            threadIdx.x + (static_cast<unsigned>(b) << BLOCK_SHIFT);
        bool in_range = thread_id < len;
        int data = in_range ? in[offset + thread_id] : 0;

        unsigned greater = in_range && data > pivot;
        unsigned gt_mask = __ballot_sync(0xffffffff, greater);
        if (gt_mask == 0) {
            greater = in_range && data >= pivot;
            gt_mask = __ballot_sync(0xffffffff, greater);
        }
        unsigned lt_mask = __ballot_sync(0xffffffff, in_range && !greater);
        unsigned gt_count = __popc(gt_mask);
        unsigned lt_count = __popc(lt_mask);

        unsigned lt_offset = 0;
        unsigned gt_offset = 0;
        if ((threadIdx.x & 31) == 0) {
            if (lt_count > 0)
                lt_offset = atomicAdd(&parts[slot].lt, lt_count);
            if (gt_count > 0)
                gt_offset = len - (atomicAdd(&parts[slot].gt, gt_count) + gt_count);
        }
        lt_offset = __shfl_sync(0xffffffff, lt_offset, 0);
        gt_offset = __shfl_sync(0xffffffff, gt_offset, 0);

        if (in_range) {
            unsigned lane_mask_lt;
            asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
            unsigned my_mask = greater ? gt_mask : lt_mask;
            unsigned my_offset = __popc(my_mask & lane_mask_lt);
            my_offset += greater ? gt_offset : lt_offset;
            out[offset + my_offset] = data;
        }
    }
}

static __device__ int alloc_part(Part *parts) {
    int slot = atomicAdd(&d_part_cursor, 1);
    if (slot >= PART_CAP) {
        atomicExch(&d_part_overflow, 1);
        return -1;
    }
    parts[slot].lt = 0;
    parts[slot].gt = 0;
    parts[slot].sorted = 0;
    __threadfence();
    return slot;
}

#pragma gtap function
__device__ void big_bitonic_task(int *data, int *dest, int *pad,
                                 unsigned offset, unsigned len) {
    big_bitonic(data, dest, pad, offset, len);
}

#pragma gtap function
__device__ void qsort_task(int *in, int *out, int *pad,
                           Part *parts, unsigned offset, unsigned len,
                           int depth, int source_is_indata, int slot) {
    if (len <= 1) {
        // The live value is in `in`. source_is_indata means `in` is already the
        // final buffer. Copy only when the result belongs in the other buffer.
        if (threadIdx.x == 0 && len == 1 && !source_is_indata)
            out[offset] = in[offset];
        return;
    }

    int *final_buf = source_is_indata ? in : out;
    if (len <= BLOCK) {
        bitonic_small(in, final_buf, offset, len);
        return;
    }

    int nblocks = static_cast<int>((len + BLOCK - 1) >> BLOCK_SHIFT);
    int task_count = nblocks < kPartitionFanout ? nblocks : kPartitionFanout;
    if (threadIdx.x == 0) {
        for (int b = 0; b < task_count; ++b) {
#pragma gtap task
            partition_block(in, out, offset, len, parts, slot, b, task_count,
                            nblocks);
        }
    }
#pragma gtap taskwait

    __shared__ unsigned s_lt;
    __shared__ unsigned s_gt;
    if (threadIdx.x == 0) {
        s_lt = parts[slot].lt;
        s_gt = parts[slot].gt;
    }
    __syncthreads();
    unsigned lt_len = s_lt;
    unsigned gt_len = s_gt;

    if (lt_len == 0) {
        if (source_is_indata) {
            for (unsigned i = threadIdx.x; i < len; i += BLOCK)
                in[offset + i] = out[offset + i];
        }
        return;
    }

    if (threadIdx.x == 0) {
        if (lt_len > BLOCK) {
            if (depth >= static_cast<int>(MAX_DEPTH)) {
#pragma gtap task
                big_bitonic_task(out, final_buf, pad, offset, lt_len);
            } else {
                int child = alloc_part(parts);
                if (child >= 0) {
#pragma gtap task
                    qsort_task(out, in, pad, parts, offset, lt_len, depth + 1,
                               source_is_indata ? 0 : 1, child);
                }
            }
        } else if (lt_len > 1) {
#pragma gtap task
            qsort_task(out, final_buf, pad, parts, offset, lt_len, depth + 1, 0,
                       slot);
        } else if (source_is_indata && lt_len == 1) {
            in[offset] = out[offset];
        }

        unsigned gt_off = offset + lt_len;
        if (gt_len > BLOCK) {
            if (depth >= static_cast<int>(MAX_DEPTH)) {
#pragma gtap task
                big_bitonic_task(out, final_buf, pad, gt_off, gt_len);
            } else {
                int child = alloc_part(parts);
                if (child >= 0) {
#pragma gtap task
                    qsort_task(out, in, pad, parts, gt_off, gt_len, depth + 1,
                               source_is_indata ? 0 : 1, child);
                }
            }
        } else if (gt_len > 1) {
#pragma gtap task
            qsort_task(out, final_buf, pad, parts, gt_off, gt_len, depth + 1, 0,
                       slot);
        } else if (source_is_indata && gt_len == 1) {
            in[gt_off] = out[gt_off];
        }
    }
}

__global__ void exec_kernel(int *data, int *scratch, int *pad,
                            Part *parts, unsigned n) {
#pragma gtap entry
    qsort_task(data, scratch, pad, parts, 0, n, 0, 1, 0);
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

    gtap_block_config config{
        .grid_size = 396,
        .block_size = static_cast<int>(BLOCK),
        .max_tasks_per_block = 20000,
        .profile_capacity_per_block = 50000,
    };
    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        return 1;
    }

    int* d_data = nullptr;
    int* d_scratch = nullptr;
    int* d_pad = nullptr;
    Part* d_parts = nullptr;
    cudaMalloc(&d_data, sizeof(int) * N);
    cudaMalloc(&d_scratch, sizeof(int) * N);
    cudaMalloc(&d_pad, sizeof(int) * N);
    cudaMalloc(&d_parts, static_cast<size_t>(PART_CAP) * sizeof(Part));
    cudaMemcpy(d_data, h.data(), sizeof(int) * N, cudaMemcpyHostToDevice);

    int cursor = 1;
    int overflow = 0;
    cudaMemcpyToSymbol(d_part_cursor, &cursor, sizeof(cursor));
    cudaMemcpyToSymbol(d_part_overflow, &overflow, sizeof(overflow));
    cudaMemset(d_parts, 0, sizeof(Part));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, d_data, d_scratch, d_pad, d_parts,
                      static_cast<unsigned>(N));
    if (err == cudaSuccess) {
        cudaEventRecord(stop);
        err = gtap_synchronize();
    }
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        cudaFree(d_scratch);
        cudaFree(d_pad);
        cudaFree(d_parts);
        gtap_finalize();
        return 1;
    }
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h.data(), d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&overflow, d_part_overflow, sizeof(overflow));
    bool ok = overflow == 0 && h == gold;
    printf("Quicksort(%zu) = %s\n", N, ok ? "Correct" : "Incorrect");
    printf("Execution time: %.3f ms\n", ms);

#ifdef GTAP_ENABLE_PROFILING
    gtap_export_profile({
        .output_directory = "./profile/quicksort_advanced_block",
        .overwrite = true,
    });
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_scratch);
    cudaFree(d_pad);
    cudaFree(d_parts);
    err = gtap_finalize();
    return ok && err == cudaSuccess ? 0 : 1;
}
