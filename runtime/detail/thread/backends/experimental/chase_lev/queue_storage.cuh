#pragma once

#include "../../../../common/cuda_primitives.cuh"
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

inline size_t queue_metadata_bytes(size_t scheduling_units, int num_queues) {
    return sizeof(WarpTaskQueueMetadata) * scheduling_units * static_cast<size_t>(num_queues);
}

inline size_t queue_slot_bytes(size_t scheduling_units, int num_queues) {
    return sizeof(int) * scheduling_units * static_cast<size_t>(num_queues) *
        static_cast<size_t>(h_launch_config.queue_capacity);
}

inline size_t queue_storage_allocation_bytes(
    size_t scheduling_units, int num_queues
) {
    return queue_metadata_bytes(scheduling_units, num_queues) +
        queue_slot_bytes(scheduling_units, num_queues);
}

// Starts the async clears. Symbols are published later.
inline cudaError_t stage_queue_storage(
    size_t scheduling_units, int num_queues,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    const size_t metadata_bytes = queue_metadata_bytes(scheduling_units, num_queues);
    const size_t slot_bytes = queue_slot_bytes(scheduling_units, num_queues);

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start);
#endif

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->metadata), metadata_bytes));

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

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->slots), slot_bytes));

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
    size_t scheduling_units, int num_queues,
    cudaStream_t stream
) {
    WarpTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_warp_task_queue_storage, sizeof(int*)));

    if (metadata != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            metadata, 0, queue_metadata_bytes(scheduling_units, num_queues), stream));
    }
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, queue_slot_bytes(scheduling_units, num_queues), stream));
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

}  // namespace gtap::detail::thread
