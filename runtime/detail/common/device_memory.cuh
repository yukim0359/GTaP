#pragma once

#include <cuda_runtime.h>

namespace gtap::detail {

// Host-side allocation and release of device memory.
// A failed cudaMalloc does not promise to leave *out null. Store the pointer
// only on success so release can free every non-null field. free_device then
// clears that field.
template <typename T>
inline cudaError_t alloc_device(T** out, size_t bytes) {
    void* ptr = nullptr;
    cudaError_t st = cudaMalloc(&ptr, bytes);
    *out = (st == cudaSuccess) ? static_cast<T*>(ptr) : nullptr;
    return st;
}

template <typename T>
inline void free_device(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

}  // namespace gtap::detail
