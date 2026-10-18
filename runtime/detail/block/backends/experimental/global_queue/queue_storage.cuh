#pragma once

#include "../../../../common/checked_size.cuh"
#include "../../../../common/device_memory.cuh"
#include "../../../../common/runtime_config.cuh"
#include "../../../../common/runtime_error.cuh"

// Depth of the per-block unpublished child-task buffer. Override with -D.
#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

namespace gtap::detail::block {

using namespace gtap::detail;

__constant__ int* d_global_task_queue; // int[num_blocks * tasks_per_block]
__device__ unsigned int d_queue_head;
__device__ unsigned int d_queue_tail;
__device__ unsigned int d_queue_alloc;
__constant__ int* d_task_id_generated; // int[num_blocks * GTAP_MAX_CHILD_TASKS]

__device__ __forceinline__ int get_task_id_generated(int block_id, int idx) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    return d_task_id_generated[offset];
}

__device__ __forceinline__ void set_task_id_generated(
    int block_id, int idx, int task_id
) {
    if (idx >= GTAP_MAX_CHILD_TASKS) {
        GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, -1, idx, GTAP_MAX_CHILD_TASKS);
    }
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    d_task_id_generated[offset] = task_id;
}

struct queue_storage_buffers {
    int* slots = nullptr;
    int* generated = nullptr;
};

inline size_t generated_task_id_bytes(const launch_config& config) {
    size_t bytes = mul_memory_bytes(
        sizeof(int), static_cast<size_t>(GTAP_MAX_CHILD_TASKS));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.total_scheduling_units));
}

inline size_t queue_slot_bytes(const launch_config& config) {
    size_t bytes = mul_memory_bytes(
        sizeof(int),
        static_cast<size_t>(config.total_scheduling_units));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.tasks_per_scheduling_unit));
}

// fixed_bytes is the unpublished-child buffer. bytes_per_task is the size of one queue slot.
inline bool queue_memory_terms(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    const size_t generated_bytes = generated_task_id_bytes(config);
    if (generated_bytes == SIZE_MAX) return false;
    *fixed_bytes = generated_bytes;
    *bytes_per_task = sizeof(int);
    return true;
}

inline size_t queue_storage_allocation_bytes(const launch_config& config) {
    return add_memory_bytes(
        queue_slot_bytes(config),
        generated_task_id_bytes(config));
}

inline cudaError_t stage_queue_storage(
    const launch_config& config,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    const size_t slot_bytes = queue_slot_bytes(config);
    const size_t generated_bytes = generated_task_id_bytes(config);
    if (slot_bytes == SIZE_MAX || generated_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->slots, slot_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, slot_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->generated, generated_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->generated, 0, generated_bytes, stream));
    return cudaSuccess;
}

inline cudaError_t publish_queue_storage(const queue_storage_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_global_task_queue, &buffers.slots, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_generated, &buffers.generated, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_queue_storage(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t slot_bytes = queue_slot_bytes(config);
    const size_t generated_bytes = generated_task_id_bytes(config);
    if (slot_bytes == SIZE_MAX || generated_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_global_task_queue, sizeof(int*)));
    int* generated = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &generated, d_task_id_generated, sizeof(int*)));
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, slot_bytes, stream));
    }
    if (generated != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            generated, 0, generated_bytes, stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_queue_storage() {
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_global_task_queue, sizeof(int*)));
    int* generated = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &generated, d_task_id_generated, sizeof(int*)));
    if (slots != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(slots));
    if (generated != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(generated));
    return cudaSuccess;
}

inline void release_staged_queue_storage(queue_storage_buffers* buffers) {
    free_device(buffers->slots);
    free_device(buffers->generated);
    int* slots = nullptr;
    int* generated = nullptr;
    cudaMemcpyToSymbol(d_global_task_queue, &slots, sizeof(slots));
    cudaMemcpyToSymbol(d_task_id_generated, &generated, sizeof(generated));
}

inline cudaError_t reset_queue_counters() {
    unsigned int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_head, &zero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_tail, &zero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_alloc, &zero, sizeof(unsigned int)));
    return cudaSuccess;
}

}  // namespace gtap::detail::block
