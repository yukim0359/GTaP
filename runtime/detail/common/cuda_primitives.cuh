#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace gtap::detail {

inline constexpr int warp_size = 32;

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

__device__ __forceinline__ unsigned long long get_global_time() {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    return time;
}

}  // namespace gtap::detail
