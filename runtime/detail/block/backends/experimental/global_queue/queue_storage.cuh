#pragma once

#include "../../../../common/runtime_config.cuh"
#include "../../../../common/runtime_error.cuh"

// Depth of the per-block unpublished child-task buffer. Override with -D.
#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

namespace gtap::detail::block {

using namespace gtap::detail;

inline constexpr int runtime_init_stream_count = 5;
inline constexpr int task_id_free_position_fill = 0xFF;

__constant__ int* d_global_task_queue;
__device__ unsigned int d_queue_head;
__device__ unsigned int d_queue_tail;
__device__ unsigned int d_queue_alloc;
__constant__ int* d_task_id_generated;

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

inline size_t queue_storage_allocation_bytes(
    size_t workers, size_t tasks, int num_queues
) {
    (void)num_queues;
    return sizeof(int) * tasks
        + sizeof(int) * workers * GTAP_MAX_CHILD_TASKS;
}

inline cudaError_t stage_queue_storage(
    size_t workers, size_t tasks, int num_queues,
    cudaStream_t streams[],
    queue_storage_buffers* buffers
) {
    (void)num_queues;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->slots), sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, sizeof(int) * tasks, streams[0]));
    const size_t generated_bytes =
        sizeof(int) * workers * GTAP_MAX_CHILD_TASKS;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->generated), generated_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->generated, 0, generated_bytes, streams[4]));
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
    size_t workers, size_t tasks, int num_queues,
    cudaStream_t streams[]
) {
    (void)num_queues;
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_global_task_queue, sizeof(int*)));
    int* generated = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &generated, d_task_id_generated, sizeof(int*)));
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, sizeof(int) * tasks, streams[0]));
    }
    if (generated != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            generated, 0, sizeof(int) * workers * GTAP_MAX_CHILD_TASKS,
            streams[4]));
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
