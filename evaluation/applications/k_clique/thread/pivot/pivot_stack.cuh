#pragma once

#include "../device_state.cuh"

enum class PivotStackField {
    Next = 0,
    Drop,
    Pivot
};

enum { kPivotStackFields = 3 };

__device__ __forceinline__ int pivot_state_levels() {
    return g_bit_buffer_count >> 1;
}

// Worker-minor: [field][level][worker]. Same (field, level) is contiguous across a warp.
struct PivotStackCol {
    int* base;
    int stride;

    __device__ __forceinline__ int& operator[](int level) const {
        return base[(size_t)level * (size_t)stride];
    }
};

__device__ __forceinline__ PivotStackCol pivot_stack(PivotStackField field) {
    int levels = pivot_state_levels();
    return {
        g_pivot_stack_scratch +
            ((size_t)(int)field * (size_t)levels * (size_t)g_num_workers) +
            (size_t)worker_id_global(),
        g_num_workers
    };
}
