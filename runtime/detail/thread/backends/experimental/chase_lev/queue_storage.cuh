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

__constant__ WarpTaskQueueMetadata** d_warp_task_queue_metadata;
__constant__ int* d_warp_task_queue_storage;

__device__ __forceinline__ int* chaselev_queue_slot(
    int queue_idx, int worker_idx, int slot
) {
    const size_t index =
        (static_cast<size_t>(queue_idx) *
             d_launch_config.total_workers +
         worker_idx) *
            d_launch_config.queue_capacity +
        slot;
    return &d_warp_task_queue_storage[index];
}

struct queue_storage_buffers {
    WarpTaskQueueMetadata** metadata = nullptr;
    WarpTaskQueueMetadata** host_planes = nullptr;
    int* slots = nullptr;
};

inline size_t queue_storage_allocation_bytes(
    size_t workers, size_t tasks, int num_queues
) {
    return sizeof(WarpTaskQueueMetadata*) * static_cast<size_t>(num_queues)
        + static_cast<size_t>(num_queues) * sizeof(WarpTaskQueueMetadata) * workers
        + sizeof(int) * tasks;
}

// Starts the async clears. Symbols are published later.
inline cudaError_t stage_queue_storage(
    size_t workers, size_t tasks, int num_queues,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    cudaEventRecord(start);
    #endif

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->metadata),
        sizeof(WarpTaskQueueMetadata*) * num_queues));

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * num_queues, elapsed);
    #endif

    buffers->host_planes = reinterpret_cast<WarpTaskQueueMetadata**>(
        malloc(sizeof(WarpTaskQueueMetadata*) * num_queues));
    for (int k = 0; k < num_queues; ++k) {
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(start);
        #endif
        WarpTaskQueueMetadata* plane_ptr = nullptr;
        GTAP_DETAIL_CUDA_TRY(cudaMalloc(
            reinterpret_cast<void**>(&plane_ptr),
            sizeof(WarpTaskQueueMetadata) * workers));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * workers, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            plane_ptr, 0, sizeof(WarpTaskQueueMetadata) * workers, stream));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemsetAsync(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * workers, elapsed);
        #endif
        buffers->host_planes[k] = plane_ptr;
    }

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(
        buffers->metadata, buffers->host_planes,
        sizeof(WarpTaskQueueMetadata*) * num_queues, cudaMemcpyHostToDevice));

    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->slots), sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, sizeof(int) * tasks, stream));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * num_queues, elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    return cudaSuccess;
}

inline cudaError_t publish_queue_storage(queue_storage_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_metadata, &buffers.metadata, sizeof(WarpTaskQueueMetadata**)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &buffers.slots, sizeof(int*)));
    free(buffers.host_planes);
    buffers.host_planes = nullptr;
    return cudaSuccess;
}

inline cudaError_t clear_queue_storage(
    size_t workers, size_t tasks, int num_queues,
    cudaStream_t stream
) {
    WarpTaskQueueMetadata** metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_warp_task_queue_storage, sizeof(int*)));

    WarpTaskQueueMetadata** host_planes = reinterpret_cast<WarpTaskQueueMetadata**>(
        malloc(sizeof(WarpTaskQueueMetadata*) * num_queues));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(
        host_planes, metadata,
        sizeof(WarpTaskQueueMetadata*) * num_queues, cudaMemcpyDeviceToHost));

    for (int k = 0; k < num_queues; ++k) {
        if (host_planes[k] != nullptr) {
            GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
                host_planes[k], 0, sizeof(WarpTaskQueueMetadata) * workers, stream));
        }
    }
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(slots, 0, sizeof(int) * tasks, stream));
    }
    free(host_planes);
    return cudaSuccess;
}

inline cudaError_t free_queue_storage() {
    const int num_queues = stored_launch_config().num_queues;
    WarpTaskQueueMetadata** metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_warp_task_queue_storage, sizeof(int*)));

    WarpTaskQueueMetadata** host_planes = reinterpret_cast<WarpTaskQueueMetadata**>(
        malloc(sizeof(WarpTaskQueueMetadata*) * num_queues));
    if (metadata != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemcpy(
            host_planes, metadata,
            sizeof(WarpTaskQueueMetadata*) * num_queues, cudaMemcpyDeviceToHost));
        for (int k = 0; k < num_queues; ++k) {
            if (host_planes[k] != nullptr) {
                GTAP_DETAIL_CUDA_TRY(cudaFree(host_planes[k]));
            }
        }
    }
    free(host_planes);

    if (metadata != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(metadata));
    if (slots != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(slots));
    return cudaSuccess;
}

}  // namespace gtap::detail::thread
