#pragma once

#include <cuda_runtime.h>
#include "gtap_runtime_error.cuh"
#include "gtap_config.cuh"

// #define DEBUG

extern const size_t __gtap_auto_task_data_size;
__constant__ size_t d_gtap_task_data_stride;

inline constexpr size_t gtap_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

inline constexpr size_t gtap_compile_time_task_data_size_limit() {
    return static_cast<size_t>(-1);
}

inline size_t gtap_host_task_data_stride() {
    return gtap_align_up(__gtap_auto_task_data_size, 16);
}

__device__ __forceinline__ size_t gtap_device_task_data_stride() {
    return d_gtap_task_data_stride;
}

inline cudaError_t gtap_init_device_task_data_stride() {
    size_t stride = gtap_host_task_data_stride();
    return cudaMemcpyToSymbol(d_gtap_task_data_stride, &stride, sizeof(size_t));
}

// Safety thresholds for error detection
#define GTAP_QUEUE_MARGIN 100
#define GTAP_TASK_ID_POOL_MIN_FREE 100  // Minimum free task IDs before overflow warning

#ifndef GTAP_CUDA_TRY
#define GTAP_CUDA_TRY(call) do { \
    cudaError_t __st = (call); \
    if (__st != cudaSuccess) { \
        if (!gtap_print_runtime_error_report()) { \
            printf("CUDA ERROR: %s\n", cudaGetErrorString(__st)); \
        } \
        return __st; \
    } \
} while (0)
#endif

template <typename SymbolT>
inline cudaError_t gtap_memset_symbol(SymbolT& symbol, int value, size_t count) {
    void* symbol_ptr = nullptr;
    cudaError_t st = cudaGetSymbolAddress(&symbol_ptr, symbol);
    if (st != cudaSuccess) return st;
    return cudaMemset(symbol_ptr, value, count);
}

template <typename SymbolT>
inline cudaError_t gtap_memset_symbol_async(
    SymbolT& symbol, int value, size_t count, cudaStream_t stream
) {
    void* symbol_ptr = nullptr;
    cudaError_t st = cudaGetSymbolAddress(&symbol_ptr, symbol);
    if (st != cudaSuccess) return st;
    return cudaMemsetAsync(symbol_ptr, value, count, stream);
}

// Termination modes
enum TerminationMode {
    TERMINATE_ON_ALL_TASKS_FINISH, // default
    TERMINATE_ON_FIRST_TASK_FINISH  // finish when first task finishes
};

__device__ __forceinline__ void __gtap_copy_bytes(void* dst, const void* src, size_t nbytes) {
    uint32_t* d32 = reinterpret_cast<uint32_t*>(dst);
    const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
    size_t n32 = nbytes / 4;
    #pragma unroll
    for (size_t i = 0; i < n32; ++i) {
        d32[i] = s32[i];
    }
    uint8_t* d8 = reinterpret_cast<uint8_t*>(d32 + n32);
    const uint8_t* s8 = reinterpret_cast<const uint8_t*>(s32 + n32);
    for (size_t i = 0; i < (nbytes - n32 * 4); ++i) {
        d8[i] = s8[i];
    }
}

// Low-level cache/ordering helpers
__device__ __forceinline__ int load_L2(int *ptr) {
    int val;
    asm volatile("ld.global.cg.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ unsigned int load_L2(unsigned int *ptr) {
    unsigned int val;
    asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ uint16_t load_L2_u16t(uint16_t *ptr) {
    uint16_t val;
    asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

__device__ __forceinline__ int load_L2_acquire(int *ptr) {
    int val;
    asm volatile("ld.global.acquire.gpu.s32 %0, [%1];\n" : "=r"(val) : "l"(ptr));
    return val;
}

// Load a pointer (64-bit) from L2 cache
__device__ __forceinline__ void* load_L2_ptr(void** ptr) {
    void* val;
    // For 64-bit pointers, use 64-bit load
    asm volatile("ld.global.cg.u64 %0, [%1];\n" : "=l"(val) : "l"(ptr));
    return val;
}

// Template specialization for int* pointers
template<typename T>
__device__ __forceinline__ T* load_L2_ptr(T** ptr) {
    return reinterpret_cast<T*>(load_L2_ptr(reinterpret_cast<void**>(ptr)));
}

__device__ __forceinline__ void store_L2(int *ptr, int val) {
    asm volatile("st.global.cg.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2(unsigned int *ptr, unsigned int val) {
    asm volatile("st.global.cg.u32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2_u16t(uint16_t *ptr, uint16_t val) {
    asm volatile("st.global.cg.u16 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2_release(int *ptr, int val) {
    asm volatile("st.global.release.gpu.s32 [%0], %1;\n" :: "l"(ptr), "r"(val));
}

__device__ __forceinline__ void store_L2_ptr(void** ptr, void* val) {
    asm volatile("st.global.cg.u64 [%0], %1;\n" :: "l"(ptr), "l"(val));
}

template<typename T>
__device__ __forceinline__ void store_L2_ptr(T** ptr, T* val) {
    store_L2_ptr(reinterpret_cast<void**>(ptr), reinterpret_cast<void*>(val));
}

// Store a structure to L2 cache using word-by-word store_L2
// This ensures the entire structure is written to L2 cache
template<typename T>
__device__ __forceinline__ void store_struct_L2(T* dst, const T& src) {
    const size_t size = sizeof(T);
    const int* src_words = reinterpret_cast<const int*>(&src);
    int* dst_words = reinterpret_cast<int*>(dst);
    const size_t num_words = size / sizeof(int);
    const size_t remaining_bytes = size % sizeof(int);
    
    // Store full words using store_L2
    for (size_t i = 0; i < num_words; ++i) {
        store_L2(&dst_words[i], src_words[i]);
    }
    
    // Handle remaining bytes (if size is not a multiple of sizeof(int))
    if (remaining_bytes != 0) {
        // For remaining bytes, use byte-by-byte copy with store_L2 for the containing word
        const unsigned char* src_bytes = reinterpret_cast<const unsigned char*>(&src);
        // Read the last word, modify the relevant bytes, then store back
        int last_word = load_L2(&dst_words[num_words]);
        unsigned char* last_word_bytes = reinterpret_cast<unsigned char*>(&last_word);
        for (size_t i = 0; i < remaining_bytes; ++i) {
            last_word_bytes[i] = src_bytes[num_words * sizeof(int) + i];
        }
        store_L2(&dst_words[num_words], last_word);
    }
}

__device__ __forceinline__ int atomicCAS_acquire(int* addr, int expected, int desired) {
    int old;
    asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(addr), "r"(expected), "r"(desired) : "memory");
    return old;
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
    return blockIdx.x * GTAP_NUM_WARPS + get_warp_id_in_block();
}

__device__ __forceinline__ int get_random_blocknum(int selfBlock) {
    unsigned int seed = (unsigned int)(clock64() + selfBlock * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int r = seed % GTAP_GRID_SIZE;
    if (r == selfBlock) r = (r + 1) % GTAP_GRID_SIZE;
    return r;
}

__device__ __forceinline__ int get_random_warpnum_global(int selfWarp) {
    unsigned int seed = (unsigned int)(clock64() + selfWarp * 2654435761u);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int totalWarps = GTAP_GRID_SIZE * GTAP_NUM_WARPS;
    int r = seed % totalWarps;
    if (r == selfWarp) r = (r + 1) % totalWarps;
    return r;
}

__device__ __forceinline__ int get_random_warpnum_in_block(int self_warp_id) {
    unsigned int seed = (unsigned int)(clock64() + self_warp_id * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int r = seed % GTAP_NUM_WARPS;
    if (r == self_warp_id) r = (r + 1) % GTAP_NUM_WARPS;
    return r;
}

__device__ __forceinline__ bool get_random_bool() {
    unsigned int seed = (unsigned int)(clock64() + threadIdx.x * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (seed % 2) == 0;
}

#ifdef PROFILE
__device__ __forceinline__ unsigned long long get_global_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}
#endif
