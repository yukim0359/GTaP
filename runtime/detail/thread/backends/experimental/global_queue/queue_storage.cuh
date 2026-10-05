#pragma once

#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/runtime_config.cuh"
#include "../../../../common/runtime_error.cuh"

// Depth of the per-queue unpublished child-task buffer. Override with -D.
#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

namespace gtap::detail::thread {

using namespace gtap::detail;

constexpr int GTAP_TASK_ID_GEN_QUEUE_STRIDE =
    GTAP_MAX_CHILD_TASKS * warp_size;

__constant__ int* d_global_task_queue;              // int[num_queues * num_warps * queue_capacity]
__constant__ int* d_queue_head;                     // int[num_queues]
__constant__ int* d_queue_tail;                     // int[num_queues]
__constant__ int* d_queue_alloc;                    // int[num_queues]
__constant__ int* d_task_id_generated_by_queue_idx; // int[num_warps * num_queues * GTAP_TASK_ID_GEN_QUEUE_STRIDE]

__device__ __forceinline__ int* global_queue_slot(
    int queue_idx, int position
) {
    const size_t capacity =
        static_cast<size_t>(d_launch_config.total_scheduling_units) *
        d_launch_config.queue_capacity;
    return &d_global_task_queue[
        static_cast<size_t>(queue_idx) * capacity + position];
}

__device__ __forceinline__ int get_task_id_generated(
    int warp_id_global, int queue_idx, int idx
) {
    int offset =
        (warp_id_global * d_launch_config.num_queues + queue_idx) *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE +
        idx;
    return d_task_id_generated_by_queue_idx[offset];
}

__device__ __forceinline__ void set_task_id_generated(
    int warp_id_global, int queue_idx, int idx, int task_id
) {
    if (idx >= GTAP_TASK_ID_GEN_QUEUE_STRIDE) {
        GTAP_DETAIL_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, queue_idx, idx, GTAP_TASK_ID_GEN_QUEUE_STRIDE);
    }
    int offset =
        (warp_id_global * d_launch_config.num_queues + queue_idx) *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE +
        idx;
    d_task_id_generated_by_queue_idx[offset] = task_id;
}

struct queue_storage_buffers {
    int* slots = nullptr;
    int* head = nullptr;
    int* tail = nullptr;
    int* alloc = nullptr;
    int* generated = nullptr;
};

inline size_t generated_task_id_bytes(size_t scheduling_units, int num_queues) {
    return sizeof(int) * scheduling_units * static_cast<size_t>(num_queues) *
        GTAP_TASK_ID_GEN_QUEUE_STRIDE;
}

inline size_t queue_slot_bytes(size_t scheduling_units, int num_queues) {
    return sizeof(int) * scheduling_units * static_cast<size_t>(num_queues) *
        static_cast<size_t>(h_launch_config.queue_capacity);
}

inline size_t queue_storage_allocation_bytes(
    size_t scheduling_units, int num_queues
) {
    return queue_slot_bytes(scheduling_units, num_queues)
        + 3 * sizeof(int) * static_cast<size_t>(num_queues)
        + generated_task_id_bytes(scheduling_units, num_queues);
}

// Starts the async clears. Symbols are published later.
inline cudaError_t stage_queue_storage(
    size_t scheduling_units, int num_queues,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    const size_t slot_bytes = queue_slot_bytes(scheduling_units, num_queues);
    const size_t metadata_bytes = sizeof(int) * static_cast<size_t>(num_queues);
    const size_t generated_bytes = generated_task_id_bytes(scheduling_units, num_queues);

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->slots), slot_bytes));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(global queue, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(buffers->slots, 0, slot_bytes, stream));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(global queue, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventRecord(start);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->generated), generated_bytes));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(task_id_generated, %zu bytes): %.3f ms\n", generated_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->generated, 0, generated_bytes, stream));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(task_id_generated, %zu bytes): %.3f ms\n", generated_bytes, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->head), metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->tail), metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->alloc), metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(buffers->head, 0, metadata_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(buffers->tail, 0, metadata_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(buffers->alloc, 0, metadata_bytes, stream));
    return cudaSuccess;
}

inline cudaError_t publish_queue_storage(const queue_storage_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_global_task_queue, &buffers.slots, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_head, &buffers.head, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_tail, &buffers.tail, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_alloc, &buffers.alloc, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_generated_by_queue_idx, &buffers.generated, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_queue_storage(
    size_t scheduling_units, int num_queues,
    cudaStream_t stream
) {
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_global_task_queue, sizeof(int*)));
    int* generated = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &generated, d_task_id_generated_by_queue_idx, sizeof(int*)));
    int* head = nullptr;
    int* tail = nullptr;
    int* alloc = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&head, d_queue_head, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&tail, d_queue_tail, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&alloc, d_queue_alloc, sizeof(int*)));

    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, queue_slot_bytes(scheduling_units, num_queues), stream));
    }
    if (generated != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            generated, 0, generated_task_id_bytes(scheduling_units, num_queues), stream));
    }
    const size_t metadata_bytes = sizeof(int) * static_cast<size_t>(num_queues);
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(head, 0, metadata_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(tail, 0, metadata_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(alloc, 0, metadata_bytes, stream));
    return cudaSuccess;
}

inline cudaError_t free_queue_storage() {
    int* slots = nullptr;
    int* head = nullptr;
    int* tail = nullptr;
    int* alloc = nullptr;
    int* generated = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_global_task_queue, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&head, d_queue_head, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&tail, d_queue_tail, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&alloc, d_queue_alloc, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &generated, d_task_id_generated_by_queue_idx, sizeof(int*)));

    if (slots != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(slots));
    if (generated != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(generated));
    if (head != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(head));
    if (tail != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(tail));
    if (alloc != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(alloc));
    return cudaSuccess;
}

}  // namespace gtap::detail::thread
