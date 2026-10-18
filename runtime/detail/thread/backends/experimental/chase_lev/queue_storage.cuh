#pragma once

#include "../../../../common/checked_size.cuh"
#include "../../../../common/cuda_primitives.cuh"
#include "../../../../common/device_memory.cuh"
#include "../../../../common/runtime_config.cuh"
#include "../../../../common/runtime_error.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

struct WarpTaskQueueMetadata {
    int top;           // Chase-Lev top (steal from here)
    int bottom;        // Chase-Lev bottom (push/pop here)
};

__constant__ WarpTaskQueueMetadata* d_warp_task_queue_metadata;  // WarpTaskQueueMetadata[num_queues * num_warps]
__constant__ int* d_warp_task_queue_storage;                     // int[num_queues * num_warps * queue_capacity]

__device__ __forceinline__ WarpTaskQueueMetadata* warp_queue_metadata(
    int queue_idx, int warp_idx
) {
    const size_t index =
        static_cast<size_t>(queue_idx) * d_launch_config.total_scheduling_units + warp_idx;
    return &d_warp_task_queue_metadata[index];
}

__device__ __forceinline__ int* chaselev_queue_slot(
    int queue_idx, int warp_idx, int slot
) {
    const size_t index =
        (static_cast<size_t>(queue_idx) *
             d_launch_config.total_scheduling_units +
         warp_idx) *
            d_launch_config.queue_capacity +
        slot;
    return &d_warp_task_queue_storage[index];
}

struct queue_storage_buffers {
    WarpTaskQueueMetadata* metadata = nullptr;
    int* slots = nullptr;
};

// fixed_bytes is the total queue metadata size, independent of task capacity.
// bytes_per_task is the size of one queue slot.
inline bool queue_memory_terms(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    size_t metadata_bytes_per_queue = 0;
    if (!checked_mul(
            sizeof(WarpTaskQueueMetadata),
            static_cast<size_t>(config.total_scheduling_units),
            &metadata_bytes_per_queue) ||
        !checked_mul(
            metadata_bytes_per_queue,
            static_cast<size_t>(config.num_queues),
            fixed_bytes)) {
        return false;
    }
    *bytes_per_task = sizeof(int);
    return true;
}

inline size_t queue_metadata_bytes(const launch_config& config) {
    size_t fixed_bytes = 0;
    size_t bytes_per_task = 0;
    if (!queue_memory_terms(config, &fixed_bytes, &bytes_per_task)) return SIZE_MAX;
    return fixed_bytes;
}

inline size_t queue_slot_bytes(const launch_config& config) {
    size_t fixed_bytes = 0;
    size_t bytes_per_task = 0;
    size_t slot_bytes = 0;
    if (!queue_memory_terms(config, &fixed_bytes, &bytes_per_task) ||
        !compute_memory_bytes(
            0,
            bytes_per_task,
            static_cast<size_t>(config.total_scheduling_units),
            static_cast<size_t>(config.tasks_per_scheduling_unit),
            &slot_bytes)) {
        return SIZE_MAX;
    }
    return slot_bytes;
}

inline size_t queue_storage_allocation_bytes(const launch_config& config) {
    return add_memory_bytes(
        queue_metadata_bytes(config),
        queue_slot_bytes(config));
}

// Starts the async clears. Symbols are published later.
inline cudaError_t stage_queue_storage(
    const launch_config& config,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    const size_t metadata_bytes = queue_metadata_bytes(config);
    const size_t slot_bytes = queue_slot_bytes(config);
    if (metadata_bytes == SIZE_MAX || slot_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

#ifdef GTAP_INTERNAL_PROFILE_INIT
    // TODO: Events leak when initialize returns early.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start);
#endif

    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->metadata, metadata_bytes));

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(queue metadata, %zu bytes): %.3f ms\n", metadata_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->metadata, 0, metadata_bytes, stream));

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(queue metadata, %zu bytes): %.3f ms\n", metadata_bytes, elapsed);
    cudaEventRecord(start);
#endif

    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->slots, slot_bytes));

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(queue slots, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventRecord(start, stream);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, slot_bytes, stream));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(queue slots, %zu bytes): %.3f ms\n", slot_bytes, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    return cudaSuccess;
}

inline cudaError_t publish_queue_storage(queue_storage_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_metadata, &buffers.metadata, sizeof(WarpTaskQueueMetadata*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &buffers.slots, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_queue_storage(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t metadata_bytes = queue_metadata_bytes(config);
    const size_t slot_bytes = queue_slot_bytes(config);
    if (metadata_bytes == SIZE_MAX || slot_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

    WarpTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_warp_task_queue_storage, sizeof(int*)));

    if (metadata != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            metadata, 0, metadata_bytes, stream));
    }
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, slot_bytes, stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_queue_storage() {
    WarpTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_warp_task_queue_storage, sizeof(int*)));

    if (metadata != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(metadata));
    if (slots != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(slots));
    return cudaSuccess;
}

inline void release_staged_queue_storage(queue_storage_buffers* buffers) {
    free_device(buffers->metadata);
    free_device(buffers->slots);
    WarpTaskQueueMetadata* metadata = nullptr;
    int* slots = nullptr;
    cudaMemcpyToSymbol(d_warp_task_queue_metadata, &metadata, sizeof(WarpTaskQueueMetadata*));
    cudaMemcpyToSymbol(d_warp_task_queue_storage, &slots, sizeof(int*));
}

}  // namespace gtap::detail::thread
