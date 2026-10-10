#pragma once

#include "../common/cuda_primitives.cuh"
#include "../common/device_memory.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

#include "task_types.cuh"

extern const size_t __gtap_auto_block_task_data_sizes[
    GTAP_MAX_THREADS_PER_BLOCK / gtap::detail::warp_size + 1];
extern const size_t __gtap_auto_task_data_align;
extern const size_t __gtap_auto_entry_result_size;

namespace gtap::detail::block {

using namespace gtap::detail;

__constant__ size_t d_task_data_stride;

inline size_t compute_task_data_stride(const launch_config& config) {
    return align_up(
        __gtap_auto_block_task_data_sizes[config.block_size / warp_size],
        __gtap_auto_task_data_align);
}

inline cudaError_t publish_task_data_stride() {
    size_t stride = compute_task_data_stride(h_launch_config);
    return cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(size_t));
}

__constant__ TaskHeader* d_task_headers;         // TaskHeader[num_blocks * tasks_per_block]
__constant__ char* d_task_data_bytes;            // char[num_blocks * tasks_per_block * task_data_stride]
__constant__ char* d_entry_result_bytes;         // char[block_size * __gtap_auto_entry_result_size]
__constant__ int* d_task_id_list_free_positions; // int[num_blocks]
__constant__ int* d_task_id_storage;             // int[num_blocks * tasks_per_block]

__global__ void init_block_id_pools_metadata() {
    if (threadIdx.x == 0) {
        d_task_id_list_free_positions[blockIdx.x] =
            d_launch_config.tasks_per_scheduling_unit;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_block_pool(
    int* id_list_free_pos,
    int* id_list_alloc_pos,
    int* id_list_free_pos_stale
) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_block = d_launch_config.tasks_per_scheduling_unit;
    int idx = old_alloc % tasks_per_block;
    int block_id = static_cast<int>(
        id_list_free_pos - d_task_id_list_free_positions);
    int id;
    bool first_use = (old_alloc < tasks_per_block);
    if (first_use) {
        id = block_id * tasks_per_block + idx;
    } else {
        id = load_L2(&d_task_id_storage[block_id * tasks_per_block + idx]);
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
        int new_free_pos = load_L2(id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(
                id, free_count, GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
        }
    }
    return id;
}

__device__ __forceinline__ void release_task_id_to_block_pool(int id) {
    const int tasks_per_block = d_launch_config.tasks_per_scheduling_unit;
    int block_id = id / tasks_per_block;
    int* id_list_free_pos = &d_task_id_list_free_positions[block_id];
    int old_free = atomicAdd(id_list_free_pos, 1);
    store_L2(
        &d_task_id_storage[
            block_id * tasks_per_block + old_free % tasks_per_block],
        id);
}

__device__ __forceinline__ void* get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * d_task_data_stride;
}

__device__ __forceinline__ void* get_entry_result_data() {
    return d_entry_result_bytes;
}

struct task_pool_buffers {
    int* id_list_free_positions = nullptr;
    int* id_storage = nullptr;
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    char* entry_result = nullptr;
};

inline size_t task_pool_allocation_bytes(const launch_config& config) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    return sizeof(int) * scheduling_units
        + sizeof(int) * tasks
        + sizeof(TaskHeader) * tasks
        + compute_task_data_stride(config) * tasks
        + __gtap_auto_entry_result_size * static_cast<size_t>(config.block_size);
}

// Allocates the pool and starts the async clears. Symbols are published later.
inline cudaError_t stage_task_pool(
    const launch_config& config,
    cudaStream_t stream,
    task_pool_buffers* buffers
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    const size_t task_data_stride = compute_task_data_stride(config);
    GTAP_DETAIL_CUDA_TRY(alloc_device(
        &buffers->id_list_free_positions, sizeof(int) * scheduling_units));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->id_list_free_positions, 0,
        sizeof(int) * scheduling_units, stream));
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->id_storage, sizeof(int) * tasks));

    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->headers, sizeof(TaskHeader) * tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->headers, 0, sizeof(TaskHeader) * tasks, stream));

    const size_t task_data_size = task_data_stride * tasks;
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->task_data, task_data_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->task_data, 0, task_data_size, stream));

    const size_t entry_result_size =
        __gtap_auto_entry_result_size * static_cast<size_t>(config.block_size);
    GTAP_DETAIL_CUDA_TRY(alloc_device(&buffers->entry_result, entry_result_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        buffers->entry_result, 0, entry_result_size, stream));
    return cudaSuccess;
}

inline cudaError_t publish_task_pool(const task_pool_buffers& buffers) {
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &buffers.id_list_free_positions,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &buffers.id_storage, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_headers, &buffers.headers, sizeof(TaskHeader*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_data_bytes, &buffers.task_data, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_entry_result_bytes, &buffers.entry_result, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(publish_task_data_stride());
    return cudaSuccess;
}

inline cudaError_t clear_task_pool(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t scheduling_units = static_cast<size_t>(config.total_scheduling_units);
    const size_t tasks =
        scheduling_units * static_cast<size_t>(config.tasks_per_scheduling_unit);
    int* id_list_free_positions = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*)));
    TaskHeader* headers = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*)));
    char* task_data = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*)));
    char* entry_result = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &entry_result, d_entry_result_bytes, sizeof(char*)));

    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            id_list_free_positions, 0,
            sizeof(int) * scheduling_units, stream));
    }
    if (headers != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            headers, 0, sizeof(TaskHeader) * tasks, stream));
    }
    if (task_data != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            task_data, 0, compute_task_data_stride(config) * tasks, stream));
    }
    if (entry_result != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            entry_result, 0,
            __gtap_auto_entry_result_size * static_cast<size_t>(config.block_size),
            stream));
    }
    return cudaSuccess;
}

inline cudaError_t free_task_pool() {
    int* id_list_free_positions = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*)));
    int* id_storage = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*)));
    TaskHeader* headers = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*)));
    char* task_data = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*)));
    char* entry_result = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &entry_result, d_entry_result_bytes, sizeof(char*)));

    if (id_list_free_positions != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(id_list_free_positions));
    }
    if (id_storage != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(id_storage));
    if (headers != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(headers));
    if (task_data != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(task_data));
    if (entry_result != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(entry_result));
    return cudaSuccess;
}

inline void release_staged_task_pool(task_pool_buffers* buffers) {
    free_device(buffers->id_list_free_positions);
    free_device(buffers->id_storage);
    free_device(buffers->headers);
    free_device(buffers->task_data);
    free_device(buffers->entry_result);
    int* id_list_free_positions = nullptr;
    int* id_storage = nullptr;
    TaskHeader* headers = nullptr;
    char* task_data = nullptr;
    char* entry_result = nullptr;
    size_t stride = 0;
    cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &id_list_free_positions,
        sizeof(id_list_free_positions));
    cudaMemcpyToSymbol(d_task_id_storage, &id_storage, sizeof(id_storage));
    cudaMemcpyToSymbol(d_task_headers, &headers, sizeof(TaskHeader*));
    cudaMemcpyToSymbol(d_task_data_bytes, &task_data, sizeof(task_data));
    cudaMemcpyToSymbol(d_entry_result_bytes, &entry_result, sizeof(entry_result));
    cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(stride));
}

}  // namespace gtap::detail::block
