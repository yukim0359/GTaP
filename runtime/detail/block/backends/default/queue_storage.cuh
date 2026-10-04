#pragma once

#include "../../../common/runtime_config.cuh"
#include "../../../common/runtime_error.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

// TODO: Init policy, not queue storage. See lifecycle.cuh.
inline constexpr int runtime_init_stream_count = 4;
inline constexpr int task_id_free_position_fill = 0;

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

inline size_t queue_storage_allocation_bytes(
    size_t scheduling_units, size_t tasks, int num_queues
) {
    (void)num_queues;
    return sizeof(BlockTaskQueueMetadata) * scheduling_units + sizeof(int) * tasks;
}

inline cudaError_t stage_queue_storage(
    size_t scheduling_units, size_t tasks, int num_queues,
    cudaStream_t streams[],
    queue_storage_buffers* buffers
) {
    (void)num_queues;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->metadata),
        sizeof(BlockTaskQueueMetadata) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->metadata, 0, sizeof(BlockTaskQueueMetadata) * scheduling_units,
        streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&buffers->slots), sizeof(int) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->slots, 0, sizeof(int) * tasks, streams[0]));
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
    size_t scheduling_units, size_t tasks, int num_queues,
    cudaStream_t streams[]
) {
    (void)num_queues;
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
            streams[0]));
    }
    if (slots != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            slots, 0, sizeof(int) * tasks, streams[0]));
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

inline cudaError_t reset_queue_counters() {
    return cudaSuccess;
}

}  // namespace gtap::detail::block
