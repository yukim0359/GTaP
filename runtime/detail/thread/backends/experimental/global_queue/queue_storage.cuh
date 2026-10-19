#pragma once

#include "../../../../common/checked_size.cuh"
#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/device_memory.cuh"
#include "../../../../common/init_error.cuh"
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

inline size_t generated_task_id_bytes(const launch_config& config) {
    size_t bytes = mul_memory_bytes(
        sizeof(int), static_cast<size_t>(config.total_scheduling_units));
    bytes = mul_memory_bytes(bytes, static_cast<size_t>(config.num_queues));
    return mul_memory_bytes(bytes, static_cast<size_t>(GTAP_TASK_ID_GEN_QUEUE_STRIDE));
}

// Bytes of one cursor array. Head, tail, and alloc each have this size.
inline size_t queue_cursor_bytes(const launch_config& config) {
    return mul_memory_bytes(sizeof(int), static_cast<size_t>(config.num_queues));
}

inline size_t queue_slot_bytes(const launch_config& config) {
    size_t bytes = mul_memory_bytes(
        sizeof(int),
        static_cast<size_t>(config.total_scheduling_units));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.tasks_per_scheduling_unit));
}

// fixed_bytes is the queue cursors and the unpublished-child buffer.
// bytes_per_task is the size of one global-queue slot.
inline bool queue_fixed_and_per_task_bytes(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    size_t cursor_bytes = queue_cursor_bytes(config);
    cursor_bytes = add_memory_bytes(cursor_bytes, queue_cursor_bytes(config));
    cursor_bytes = add_memory_bytes(cursor_bytes, queue_cursor_bytes(config));
    const size_t generated_bytes = generated_task_id_bytes(config);
    const size_t fixed = add_memory_bytes(cursor_bytes, generated_bytes);
    if (fixed == SIZE_MAX) return false;
    *fixed_bytes = fixed;
    *bytes_per_task = sizeof(int);
    return true;
}

inline size_t queue_storage_allocation_bytes(const launch_config& config) {
    const size_t cursor_bytes = queue_cursor_bytes(config);
    size_t bytes = queue_slot_bytes(config);
    bytes = add_memory_bytes(bytes, generated_task_id_bytes(config));
    bytes = add_memory_bytes(bytes, cursor_bytes);
    bytes = add_memory_bytes(bytes, cursor_bytes);
    return add_memory_bytes(bytes, cursor_bytes);
}

// Starts the async clears. Symbols are published later.
inline cudaError_t stage_queue_storage(
    const launch_config& config,
    cudaStream_t stream,
    queue_storage_buffers* buffers,
    init_error& error
) {
    const size_t slot_bytes = queue_slot_bytes(config);
    const size_t generated_bytes = generated_task_id_bytes(config);
    const size_t cursor_bytes = queue_cursor_bytes(config);
    if (slot_bytes == SIZE_MAX || generated_bytes == SIZE_MAX ||
        cursor_bytes == SIZE_MAX) {
        return error.fail(cudaErrorInvalidValue, "queue storage size overflowed");
    }

#ifdef GTAP_INTERNAL_PROFILE_INIT
    // TODO: Events leak when initialize returns early.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start);
#endif

    cudaError_t status = check_init(
        error,
        alloc_device(&buffers->slots, slot_bytes),
        "failed to allocate queue storage (allocation size: %zu bytes)",
        slot_bytes);
    if (status != cudaSuccess) return status;
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(global queue, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif
    status = check_init(
        error,
        cudaMemsetAsync(buffers->slots, 0, slot_bytes, stream),
        "failed to initialize queue storage");
    if (status != cudaSuccess) return status;
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(global queue, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventRecord(start);
#endif

    status = check_init(
        error,
        alloc_device(&buffers->generated, generated_bytes),
        "failed to allocate queue storage (allocation size: %zu bytes)",
        generated_bytes);
    if (status != cudaSuccess) return status;
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(task_id_generated, %zu bytes): %.3f ms\n", generated_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif
    status = check_init(
        error,
        cudaMemsetAsync(buffers->generated, 0, generated_bytes, stream),
        "failed to initialize queue storage");
    if (status != cudaSuccess) return status;
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(task_id_generated, %zu bytes): %.3f ms\n", generated_bytes, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    status = check_init(
        error,
        alloc_device(&buffers->head, cursor_bytes),
        "failed to allocate queue storage (allocation size: %zu bytes)",
        cursor_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        alloc_device(&buffers->tail, cursor_bytes),
        "failed to allocate queue storage (allocation size: %zu bytes)",
        cursor_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        alloc_device(&buffers->alloc, cursor_bytes),
        "failed to allocate queue storage (allocation size: %zu bytes)",
        cursor_bytes);
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        cudaMemsetAsync(buffers->head, 0, cursor_bytes, stream),
        "failed to initialize queue storage");
    if (status != cudaSuccess) return status;
    status = check_init(
        error,
        cudaMemsetAsync(buffers->tail, 0, cursor_bytes, stream),
        "failed to initialize queue storage");
    if (status != cudaSuccess) return status;
    return check_init(
        error,
        cudaMemsetAsync(buffers->alloc, 0, cursor_bytes, stream),
        "failed to initialize queue storage");
}

inline cudaError_t publish_queue_storage(const queue_storage_buffers& buffers) {
    cudaError_t status = cudaMemcpyToSymbol(
        d_global_task_queue, &buffers.slots, sizeof(int*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_queue_head, &buffers.head, sizeof(int*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_queue_tail, &buffers.tail, sizeof(int*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_queue_alloc, &buffers.alloc, sizeof(int*));
    if (status != cudaSuccess) return status;
    return cudaMemcpyToSymbol(
        d_task_id_generated_by_queue_idx, &buffers.generated, sizeof(int*));
}

inline cudaError_t clear_queue_storage(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t slot_bytes = queue_slot_bytes(config);
    const size_t generated_bytes = generated_task_id_bytes(config);
    const size_t cursor_bytes = queue_cursor_bytes(config);
    if (slot_bytes == SIZE_MAX || generated_bytes == SIZE_MAX ||
        cursor_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

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
            slots, 0, slot_bytes, stream));
    }
    if (generated != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            generated, 0, generated_bytes, stream));
    }
    if (head != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(head, 0, cursor_bytes, stream));
    }
    if (tail != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(tail, 0, cursor_bytes, stream));
    }
    if (alloc != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(alloc, 0, cursor_bytes, stream));
    }
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

inline void release_staged_queue_storage(queue_storage_buffers* buffers) {
    free_device(buffers->slots);
    free_device(buffers->head);
    free_device(buffers->tail);
    free_device(buffers->alloc);
    free_device(buffers->generated);
    int* slots = nullptr;
    int* head = nullptr;
    int* tail = nullptr;
    int* alloc = nullptr;
    int* generated = nullptr;
    cudaMemcpyToSymbol(d_global_task_queue, &slots, sizeof(slots));
    cudaMemcpyToSymbol(d_queue_head, &head, sizeof(head));
    cudaMemcpyToSymbol(d_queue_tail, &tail, sizeof(tail));
    cudaMemcpyToSymbol(d_queue_alloc, &alloc, sizeof(alloc));
    cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx, &generated, sizeof(generated));
}

}  // namespace gtap::detail::thread
