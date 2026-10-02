#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include "gtap_runtime_error.cuh"

#define GTAP_WARP_SIZE 32
#define GTAP_MAX_THREADS_PER_BLOCK 1024

// #define GTAP_DETAIL_INTERNAL_DEBUG

// Safety thresholds for error detection
#define GTAP_DETAIL_QUEUE_MARGIN 100
#define GTAP_DETAIL_TASK_ID_POOL_MIN_FREE 100

#ifndef GTAP_DETAIL_CUDA_TRY
#define GTAP_DETAIL_CUDA_TRY(call) do { \
    cudaError_t __st = (call); \
    if (__st != cudaSuccess) { \
        if (!gtap::detail::print_runtime_error_report()) { \
            printf("CUDA ERROR: %s\n", cudaGetErrorString(__st)); \
        } \
        return __st; \
    } \
} while (0)
#endif

namespace gtap::detail {

__constant__ size_t d_task_data_stride;

struct launch_config {
    int grid_size;
    int block_size;
    int warps_per_block;
    int total_workers;
    int tasks_per_worker;
    int num_queues;
    int queue_capacity;
    int profile_interval_capacity;
    size_t dynamic_shared_bytes;
};

__constant__ launch_config d_launch_config;

enum TerminationMode {
    TERMINATE_ON_ALL_TASKS_FINISH, // default
    TERMINATE_ON_FIRST_TASK_FINISH  // finish when first task finishes
};

inline launch_config& stored_launch_config() {
    static launch_config config{
        1024,
        256,
        8,
        8192,
        0,
        1,
        0,
        15000,
        0
    };
    return config;
}

__host__ __device__ __forceinline__ int profile_capacity() {
#ifdef __CUDA_ARCH__
    return 2 * d_launch_config.profile_interval_capacity;
#else
    return 2 * stored_launch_config().profile_interval_capacity;
#endif
}

inline cudaStream_t& stored_stream() {
    static cudaStream_t stream = nullptr;
    return stream;
}

inline bool& initialized_flag() {
    static bool initialized = false;
    return initialized;
}

inline cudaError_t publish_launch_config(const launch_config& config) {
    stored_launch_config() = config;
    return cudaMemcpyToSymbol(d_launch_config, &config, sizeof(config));
}

__host__ __device__ inline constexpr size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Low-level cache/ordering helpers
__device__ __forceinline__ uint16_t load_L2(uint16_t *ptr) {
    unsigned int val;
    asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return static_cast<uint16_t>(val);
}

__device__ __forceinline__ unsigned int load_L2(unsigned int *ptr) {
    unsigned int val;
    asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ int load_L2(int *ptr) {
    int val;
    asm volatile("ld.global.cg.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ void* load_L2(void** ptr) {
    void* val;
    asm volatile("ld.global.cg.u64 %0, [%1];\n" : "=l"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ int load_L2_acquire(int *ptr) {
    int val;
    asm volatile("ld.global.acquire.gpu.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ void store_L2(uint16_t *ptr, uint16_t val) {
    unsigned int wide = val;
    asm volatile("st.global.cg.u16 [%0], %1;\n" :: "l"(ptr), "r"(wide));
}

__device__ __forceinline__ void store_L2(unsigned int *ptr, unsigned int val) {
    asm volatile("st.global.cg.u32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2(int *ptr, int val) {
    asm volatile("st.global.cg.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2(void** ptr, void* val) {
    asm volatile("st.global.cg.u64 [%0], %1;\n" :: "l"(ptr), "l"(val));
}

__device__ __forceinline__ void store_L2_release(int *ptr, int val) {
    asm volatile("st.global.release.gpu.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void prefetch_global_L2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr) : "memory");
}

__device__ __forceinline__ void lock(int* lock_var) {
    while (atomicCAS(lock_var, 0, 1) != 0) {}
}

__device__ __forceinline__ void unlock(int* lock_var) {
    atomicExch(lock_var, 0);
}

__device__ __forceinline__ unsigned int get_lane_id() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ unsigned int get_warp_id_in_block() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ unsigned int get_warp_id_global() {
    return blockIdx.x * d_launch_config.warps_per_block + get_warp_id_in_block();
}

__device__ __forceinline__ int get_random_block_id(int selfBlock) {
    unsigned int seed = (unsigned int)(clock64() + selfBlock * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int totalBlocks = d_launch_config.grid_size;
    int r = seed % totalBlocks;
    if (r == selfBlock) r = (r + 1) % totalBlocks;
    return r;
}

__device__ __forceinline__ int get_random_warp_id_global(int selfWarp) {
    unsigned int seed = (unsigned int)(clock64() + selfWarp * 2654435761u);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int totalWarps = d_launch_config.total_workers;
    int r = seed % totalWarps;
    if (r == selfWarp) r = (r + 1) % totalWarps;
    return r;
}

#ifdef GTAP_ENABLE_PROFILING
__device__ __forceinline__ unsigned long long get_global_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}
#endif

}  // namespace gtap::detail

template<class Kernel, class... Args>
inline cudaError_t gtap_launch(Kernel kernel, Args&&... args) {
    if (!gtap::detail::initialized_flag()) {
        return cudaErrorInitializationError;
    }
    const gtap::detail::launch_config& config = gtap::detail::stored_launch_config();
    if constexpr (sizeof...(Args) == 0) {
        return cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(static_cast<unsigned int>(config.grid_size)),
            dim3(static_cast<unsigned int>(config.block_size)),
            nullptr,
            config.dynamic_shared_bytes,
            gtap::detail::stored_stream()
        );
    } else {
        void* packed_arguments[] = {
            const_cast<void*>(
                static_cast<const void*>(std::addressof(args))
            )...
        };
        return cudaLaunchKernel(
            reinterpret_cast<const void*>(kernel),
            dim3(static_cast<unsigned int>(config.grid_size)),
            dim3(static_cast<unsigned int>(config.block_size)),
            packed_arguments,
            config.dynamic_shared_bytes,
            gtap::detail::stored_stream()
        );
    }
}
