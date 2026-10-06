#pragma once

#include "../../../common/device_memory.cuh"
#include "../../../common/runtime_config.cuh"
#include "../../../common/runtime_error.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

struct BlockTaskQueueMetadata {
    int top;
    int bottom;
};

__constant__ BlockTaskQueueMetadata* d_block_task_queue_metadata; // BlockTaskQueueMetadata[num_blocks]
__constant__ int* d_block_task_queue_storage;                     // int[num_blocks * queue_capacity]

__device__ __forceinline__ int* block_queue_slot(
    int block_idx, int slot
) {
    return &d_block_task_queue_storage[
        static_cast<size_t>(block_idx) *
            d_launch_config.queue_capacity + slot];
}

struct queue_storage_buffers {
    BlockTaskQueueMetadata* metadata = nullptr;
    int* slots = nullptr;
};

inline size_t queue_storage_allocation_bytes(const launch_config& config) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    return sizeof(BlockTaskQueueMetadata) * scheduling_units + sizeof(int) * tasks;
}

inline cudaError_t stage_queue_storage(
    const launch_config& config,
    cudaStream_t stream,
    queue_storage_buffers* buffers
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    GTAP_DETAIL_CUDA_TRY(alloc_device(
        &buffers->metadata, sizeof(BlockTaskQueueMetadata) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->metadata, 0, sizeof(BlockTaskQueueMetadata) * scheduling_units,
        stream));
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->slots, sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, sizeof(int) * tasks, stream));
    return cudaSuccess;
}

inline cudaError_t publish_queue_storage(const queue_storage_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_block_task_queue_metadata, &buffers.metadata,
        sizeof(BlockTaskQueueMetadata*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_block_task_queue_storage, &buffers.slots, sizeof(int*)));
    return cudaSuccess;
}

inline cudaError_t clear_queue_storage(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    BlockTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_block_task_queue_metadata,
        sizeof(BlockTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_block_task_queue_storage, sizeof(int*)));
    if (metadata != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            metadata, 0, sizeof(BlockTaskQueueMetadata) * scheduling_units,
            stream));
    }
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, sizeof(int) * tasks, stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_queue_storage() {
    BlockTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_block_task_queue_metadata,
        sizeof(BlockTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_block_task_queue_storage, sizeof(int*)));
    if (metadata != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(metadata));
    if (slots != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(slots));
    return cudaSuccess;
}

inline void release_staged_queue_storage(queue_storage_buffers* buffers) {
    free_device(buffers->metadata);
    free_device(buffers->slots);
    BlockTaskQueueMetadata* metadata = nullptr;
    int* slots = nullptr;
    cudaMemcpyToSymbol(d_block_task_queue_metadata, &metadata, sizeof(BlockTaskQueueMetadata*));
    cudaMemcpyToSymbol(d_block_task_queue_storage, &slots, sizeof(int*));
}

inline cudaError_t reset_queue_counters() {
    return cudaSuccess;
}

}  // namespace gtap::detail::block
