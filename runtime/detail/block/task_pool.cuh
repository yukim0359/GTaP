#pragma once

#include "../common/checked_size.cuh"
#include "../common/cuda_primitives.cuh"
#include "../common/device_memory.cuh"
#include "../common/host_api_error.cuh"
#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"
#include "../common/task_pool.cuh"

#include "task_types.cuh"
#ifdef GTAP_ENABLE_PROFILING
#include "profile_buffer.cuh"
#endif

// The compiler sets these strides after rounding the task records up to their alignment.
// The index is the number of warps in the block.
extern const size_t __gtap_block_task_data_strides[
    GTAP_DETAIL_MAX_THREADS_PER_BLOCK / gtap::detail::warp_size + 1];
extern const size_t __gtap_entry_result_size;

namespace gtap::detail::block {

using namespace gtap::detail;

__constant__ size_t d_task_data_stride;

inline size_t compute_task_data_stride(const launch_config& config) {
    return __gtap_block_task_data_strides[config.block_size / warp_size];
}

inline cudaError_t publish_task_data_stride() {
    size_t stride = compute_task_data_stride(h_launch_config);
    return cudaMemcpyToSymbol(d_task_data_stride, &stride, sizeof(size_t));
}

__constant__ TaskHeader* d_task_headers;         // TaskHeader[num_blocks * tasks_per_block]
__constant__ char* d_task_data_bytes;            // char[num_blocks * tasks_per_block * task_data_stride]
__constant__ char* d_entry_result_bytes;         // char[block_size * __gtap_entry_result_size]
__constant__ int* d_task_id_list_free_positions; // int[num_blocks]
// High bit is the ring lap. The low 31 bits are the task id.
__constant__ int* d_task_id_storage;             // int[num_blocks * tasks_per_block]

// Launched as <<<grid_size, 1>>>.
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
    const int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    const int tasks_per_block = d_launch_config.tasks_per_scheduling_unit;
#ifdef GTAP_ENABLE_PROFILING
    // TODO: A return from another block can land between the alloc_pos update
    // and this free_pos load, so the sampled count can miss the peak. atomicMax
    // cannot recover a value that was never recorded. An exact peak needs the
    // allocation and the return to update one consistent count.
    const int unreturned_id_count =
        (old_alloc + 1) + tasks_per_block - load_L2(id_list_free_pos);
    record_unreturned_id_count(blockIdx.x, unreturned_id_count);
#endif
    const bool first_use = (old_alloc < tasks_per_block);
    int task_id = 0;
    if (first_use) {
        task_id = blockIdx.x * tasks_per_block + old_alloc;
    } else {
        const int storage_idx = blockIdx.x * tasks_per_block + old_alloc % tasks_per_block;
        const int packed = load_relaxed(&d_task_id_storage[storage_idx]);
        if (ring_lap_matches(packed, old_alloc, tasks_per_block)) {
            task_id = unpack_task_id_slot(packed);
        } else {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_SLOT_BUSY(task_id, old_alloc, tasks_per_block);
        }
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
        const int new_free_pos = load_L2(id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < GTAP_DETAIL_TASK_ID_POOL_MIN_FREE) {
            GTAP_DETAIL_RECORD_TASK_ID_POOL_LOW_HEADROOM(
                task_id, free_count, GTAP_DETAIL_TASK_ID_POOL_MIN_FREE);
        }
    }
    return task_id;
}

__device__ __forceinline__ void release_task_id_to_block_pool(int tid) {
    const int tasks_per_block = d_launch_config.tasks_per_scheduling_unit;
    const int return_block_id = tid / tasks_per_block;
    int* const id_list_free_pos = &d_task_id_list_free_positions[return_block_id];
    const int old_free = atomicAdd(id_list_free_pos, 1);
    const int storage_idx =
        return_block_id * tasks_per_block + old_free % tasks_per_block;
    store_relaxed(
        &d_task_id_storage[storage_idx],
        pack_task_id_slot(tid, old_free, tasks_per_block));
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

inline size_t free_position_bytes(const launch_config& config) {
    return mul_memory_bytes(
        sizeof(int),
        static_cast<size_t>(config.total_scheduling_units));
}

inline size_t entry_result_bytes(const launch_config& config) {
    return mul_memory_bytes(
        __gtap_entry_result_size,
        static_cast<size_t>(config.block_size));
}

// Total bytes for one task-pool component across all task slots.
inline size_t task_pool_component_bytes(
    const launch_config& config,
    size_t bytes_per_task
) {
    size_t bytes = mul_memory_bytes(
        bytes_per_task,
        static_cast<size_t>(config.total_scheduling_units));
    return mul_memory_bytes(
        bytes, static_cast<size_t>(config.tasks_per_scheduling_unit));
}

// fixed_bytes is the free-position array and the entry result.
// bytes_per_task is the task record, its header, and one task-id pool slot.
inline bool task_pool_fixed_and_per_task_bytes(
    const launch_config& config,
    size_t* fixed_bytes,
    size_t* bytes_per_task
) {
    const size_t fixed = add_memory_bytes(
        free_position_bytes(config),
        entry_result_bytes(config));
    size_t id_and_header = 0;
    size_t per_task = 0;
    if (fixed == SIZE_MAX ||
        !checked_add(sizeof(int), sizeof(TaskHeader), &id_and_header) ||
        !checked_add(id_and_header, compute_task_data_stride(config), &per_task)) {
        return false;
    }
    *fixed_bytes = fixed;
    *bytes_per_task = per_task;
    return true;
}

inline size_t task_pool_allocation_bytes(const launch_config& config) {
    size_t bytes = free_position_bytes(config);
    bytes = add_memory_bytes(
        bytes, task_pool_component_bytes(config, sizeof(int)));
    bytes = add_memory_bytes(
        bytes, task_pool_component_bytes(config, sizeof(TaskHeader)));
    bytes = add_memory_bytes(
        bytes, task_pool_component_bytes(config, compute_task_data_stride(config)));
    return add_memory_bytes(bytes, entry_result_bytes(config));
}

// Allocates the pool and starts the async clears. Symbols are published later.
inline cudaError_t stage_task_pool(
    const launch_config& config,
    cudaStream_t stream,
    task_pool_buffers* buffers,
    host_api_error& error
) {
    const size_t free_positions_bytes = free_position_bytes(config);
    const size_t id_storage_bytes = task_pool_component_bytes(config, sizeof(int));
    const size_t header_bytes = task_pool_component_bytes(config, sizeof(TaskHeader));
    const size_t task_data_bytes = task_pool_component_bytes(config, compute_task_data_stride(config));
    const size_t entry_result_size_bytes = entry_result_bytes(config);
    if (free_positions_bytes == SIZE_MAX || id_storage_bytes == SIZE_MAX ||
        header_bytes == SIZE_MAX || task_data_bytes == SIZE_MAX ||
        entry_result_size_bytes == SIZE_MAX) {
        return error.fail(cudaErrorInvalidValue, "task storage size overflowed");
    }

    cudaError_t status = record_error_if_failed(
        error,
        alloc_device(&buffers->id_list_free_positions, free_positions_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        free_positions_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->id_list_free_positions, 0, free_positions_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        alloc_device(&buffers->id_storage, id_storage_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        id_storage_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->id_storage, 0, id_storage_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        alloc_device(&buffers->headers, header_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        header_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->headers, 0, header_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        alloc_device(&buffers->task_data, task_data_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        task_data_bytes);
    if (status != cudaSuccess) return status;
    status = record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->task_data, 0, task_data_bytes, stream),
        "failed to initialize task storage");
    if (status != cudaSuccess) return status;

    status = record_error_if_failed(
        error,
        alloc_device(&buffers->entry_result, entry_result_size_bytes),
        "failed to allocate task storage (allocation size: %zu bytes)",
        entry_result_size_bytes);
    if (status != cudaSuccess) return status;
    return record_error_if_failed(
        error,
        cudaMemsetAsync(buffers->entry_result, 0, entry_result_size_bytes, stream),
        "failed to initialize task storage");
}

inline cudaError_t publish_task_pool(const task_pool_buffers& buffers) {
    cudaError_t status = cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &buffers.id_list_free_positions, sizeof(int*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_task_id_storage, &buffers.id_storage, sizeof(int*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_task_headers, &buffers.headers, sizeof(TaskHeader*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(d_task_data_bytes, &buffers.task_data, sizeof(char*));
    if (status != cudaSuccess) return status;
    status = cudaMemcpyToSymbol(
        d_entry_result_bytes, &buffers.entry_result, sizeof(char*));
    if (status != cudaSuccess) return status;
    return publish_task_data_stride();
}

inline cudaError_t clear_task_pool(
    const launch_config& config,
    cudaStream_t stream
) {
    const size_t free_positions_bytes = free_position_bytes(config);
    const size_t id_storage_bytes = task_pool_component_bytes(config, sizeof(int));
    const size_t header_bytes = task_pool_component_bytes(config, sizeof(TaskHeader));
    const size_t task_data_bytes = task_pool_component_bytes(config, compute_task_data_stride(config));
    const size_t entry_result_size_bytes = entry_result_bytes(config);
    if (free_positions_bytes == SIZE_MAX || id_storage_bytes == SIZE_MAX ||
        header_bytes == SIZE_MAX || task_data_bytes == SIZE_MAX ||
        entry_result_size_bytes == SIZE_MAX) {
        return cudaErrorInvalidValue;
    }

    int* id_list_free_positions = nullptr;
    cudaError_t status = cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*));
    if (status != cudaSuccess) return status;
    int* id_storage = nullptr;
    status = cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*));
    if (status != cudaSuccess) return status;
    TaskHeader* headers = nullptr;
    status = cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*));
    if (status != cudaSuccess) return status;
    char* task_data = nullptr;
    status = cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*));
    if (status != cudaSuccess) return status;
    char* entry_result = nullptr;
    status = cudaMemcpyFromSymbol(
        &entry_result, d_entry_result_bytes, sizeof(char*));
    if (status != cudaSuccess) return status;

    if (id_list_free_positions != nullptr) {
        status = cudaMemsetAsync(
            id_list_free_positions, 0, free_positions_bytes, stream);
        if (status != cudaSuccess) return status;
    }
    if (id_storage != nullptr) {
        status = cudaMemsetAsync(id_storage, 0, id_storage_bytes, stream);
        if (status != cudaSuccess) return status;
    }
    if (headers != nullptr) {
        status = cudaMemsetAsync(headers, 0, header_bytes, stream);
        if (status != cudaSuccess) return status;
    }
    if (task_data != nullptr) {
        status = cudaMemsetAsync(task_data, 0, task_data_bytes, stream);
        if (status != cudaSuccess) return status;
    }
    if (entry_result != nullptr) {
        status = cudaMemsetAsync(
            entry_result, 0, entry_result_size_bytes, stream);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}

inline cudaError_t free_task_pool() {
    cudaError_t result = cudaSuccess;

    int* id_list_free_positions = nullptr;
    cudaError_t free_positions_read_status = cudaMemcpyFromSymbol(
        &id_list_free_positions, d_task_id_list_free_positions, sizeof(int*));
    keep_first_error(free_positions_read_status, &result);
    int* id_storage = nullptr;
    cudaError_t id_storage_read_status = cudaMemcpyFromSymbol(
        &id_storage, d_task_id_storage, sizeof(int*));
    keep_first_error(id_storage_read_status, &result);
    TaskHeader* headers = nullptr;
    cudaError_t headers_read_status = cudaMemcpyFromSymbol(
        &headers, d_task_headers, sizeof(TaskHeader*));
    keep_first_error(headers_read_status, &result);
    char* task_data = nullptr;
    cudaError_t task_data_read_status = cudaMemcpyFromSymbol(
        &task_data, d_task_data_bytes, sizeof(char*));
    keep_first_error(task_data_read_status, &result);
    char* entry_result = nullptr;
    cudaError_t entry_result_read_status = cudaMemcpyFromSymbol(
        &entry_result, d_entry_result_bytes, sizeof(char*));
    keep_first_error(entry_result_read_status, &result);

    if (free_positions_read_status == cudaSuccess && id_list_free_positions != nullptr) {
        keep_first_error(cudaFree(id_list_free_positions), &result);
    }
    if (id_storage_read_status == cudaSuccess && id_storage != nullptr) {
        keep_first_error(cudaFree(id_storage), &result);
    }
    if (headers_read_status == cudaSuccess && headers != nullptr) {
        keep_first_error(cudaFree(headers), &result);
    }
    if (task_data_read_status == cudaSuccess && task_data != nullptr) {
        keep_first_error(cudaFree(task_data), &result);
    }
    if (entry_result_read_status == cudaSuccess && entry_result != nullptr) {
        keep_first_error(cudaFree(entry_result), &result);
    }
    return result;
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
