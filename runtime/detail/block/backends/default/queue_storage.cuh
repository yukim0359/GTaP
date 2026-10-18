#pragma once

#include "../../../common/checked_size.cuh"
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

inline size_t queue_metadata_bytes(const launch_config& config) {
    return mul_memory_bytes(
        sizeof(BlockTaskQueueMetadata),
        static_cast<size_t>(config.total_scheduling_units));
}

inline size_t queue_slot_bytes(const launch_config& config) {
    size_t bytes = mul_memory_bytes(
        sizeof(int),
        static_cast<size_t>(config.total_scheduling_units));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.tasks_per_scheduling_unit));
}

// fixed_bytes is the metadata array, independent of task capacity.
// bytes_per_task is the size of one queue slot.
inline bool queue_fixed_and_per_task_bytes(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    const size_t metadata_bytes = queue_metadata_bytes(config);
    if (metadata_bytes == SIZE_MAX) return false;
    *fixed_bytes = metadata_bytes;
    *bytes_per_task = sizeof(int);
    return true;
}

inline size_t queue_storage_allocation_bytes(const launch_config& config) {
    return add_memory_bytes(
        queue_metadata_bytes(config),
        queue_slot_bytes(config));
}

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
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->metadata, metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->metadata, 0, metadata_bytes, stream));
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->slots, slot_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, slot_bytes, stream));
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
    const size_t metadata_bytes = queue_metadata_bytes(config);
    const size_t slot_bytes = queue_slot_bytes(config);
    if (metadata_bytes == SIZE_MAX || slot_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

    BlockTaskQueueMetadata* metadata = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &metadata, d_block_task_queue_metadata,
        sizeof(BlockTaskQueueMetadata*)));
    int* slots = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &slots, d_block_task_queue_storage, sizeof(int*)));
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
