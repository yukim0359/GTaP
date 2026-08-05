#pragma once

#include <cuda_runtime.h>
#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_BLOCK
#define __GTAP_WORKER_IS_BLOCK
#endif

#include "gtap_block_core.cuh"

#ifndef GTAP_BLOCK_SIZE
#error "GTAP block mode requires GTAP_BLOCK_SIZE at compile time"
#endif

struct BlockTaskQueue {
    int top;
    int bottom;
};

struct gtap_block_config {
    int grid_size = 1024;
    int max_tasks_per_block = 100000;
    int profile_capacity_per_block = 30000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};

inline constexpr int gtap_compiled_block_size = GTAP_BLOCK_SIZE;

inline cudaError_t gtap_validate_config(const gtap_block_config& config) {
    if (config.grid_size <= 0 ||
        gtap_compiled_block_size <= 0 ||
        gtap_compiled_block_size > GTAP_MAX_THREADS_PER_BLOCK) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_block <= 0) {
        return cudaErrorInvalidValue;
    }
#ifdef GTAP_PROFILE
    if (config.profile_capacity_per_block <= 0) {
        return cudaErrorInvalidValue;
    }
#endif
    return cudaSuccess;
}

// Exposed device globals
// Note: d_task_data_bytes is now char* (byte array) to support type-erased task data (static allocation)
__constant__ BlockTaskQueue* d_block_task_queues;
__constant__ int* d_block_task_queue_storage;

__device__ __forceinline__ int* gtap_block_queue_slot(
    int block_idx, int slot
) {
    return &d_block_task_queue_storage[
        static_cast<size_t>(block_idx) *
            d_gtap_launch_config.queue_capacity + slot];
}

static size_t __gtap_runtime_device_allocation_bytes() {
    const gtap_launch_config& c = gtap_stored_launch_config();
    const size_t workers = c.total_workers;
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t queue_metadata_bytes = sizeof(BlockTaskQueue) * workers;
    const size_t queue_storage_bytes = sizeof(int) * tasks;
    const size_t task_id_list_bytes = sizeof(TaskIdList) * workers;
    const size_t task_id_storage_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = gtap_host_task_data_stride() * tasks;
    size_t total =
        queue_metadata_bytes + queue_storage_bytes + task_id_list_bytes +
        task_id_storage_bytes + header_bytes + task_data_bytes;
#ifdef GTAP_PROFILE
    total += 2 * sizeof(long long) * workers * gtap_profile_capacity();
#endif
    return total;
}

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int task_id) {
    BlockTaskQueue* q = &d_block_task_queues[blockIdx.x];
    int old_tail = atomicAdd(&ctx->queue_tail, 1);
    int top = load_L2(&q->top);
    const int queue_capacity = d_gtap_launch_config.queue_capacity;
    if (old_tail + 1 - top > queue_capacity - GTAP_QUEUE_MARGIN) {
        GTAP_RECORD_QUEUE_OVERFLOW(
            task_id, -1, old_tail + 1 - top, queue_capacity - GTAP_QUEUE_MARGIN);
    }
    store_L2(
        gtap_block_queue_slot(blockIdx.x, old_tail % queue_capacity), task_id);
    atomicAdd(&ctx->task_id_generated_count, 1);
}

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_block_task_queues_ptr), sizeof(BlockTaskQueue) * total_workers));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_block_task_queues_ptr, 0, sizeof(BlockTaskQueue) * total_workers, streams[0]));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_block_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_block_task_queue_storage_ptr, 0, sizeof(int) * total_tasks,
        streams[0]));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * total_workers));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_task_id_lists_ptr, 0, sizeof(TaskIdList) * total_workers,
        streams[1]));
    int* d_task_id_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * total_tasks));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks, streams[2]));

    // Allocate static storage for task data (type-erased as byte array)
    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = gtap_host_task_data_stride();
    size_t task_data_size = max_task_size * total_tasks;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_block_task_queues, &d_block_task_queues_ptr, sizeof(BlockTaskQueue*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_block_task_queue_storage, &d_block_task_queue_storage_ptr,
        sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_CUDA_TRY(gtap_init_device_task_data_stride());
    
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    const size_t profile_bytes =
        sizeof(long long) * total_workers * gtap_profile_capacity();
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&having_task_time_ptr), profile_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_bytes));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        having_task_time, &having_task_time_ptr, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        having_task_time_ptr, 0, profile_bytes, streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_bytes, streams[1]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[1]));
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    return cudaDeviceSynchronize();
}

cudaError_t __gtap_finalize_task_runtime() {
    // Get device pointers from symbols
    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queues_ptr, d_block_task_queues, sizeof(BlockTaskQueue*)));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_block_task_queue_storage_ptr, d_block_task_queue_storage,
        sizeof(int*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    int* d_task_id_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
#endif
    
    // Free allocated memory
    if (d_block_task_queues_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_block_task_queues_ptr));
    }
    if (d_block_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_block_task_queue_storage_ptr));
    }
    
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
#ifdef GTAP_PROFILE
    if (having_task_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(having_task_time_ptr));
    if (working_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(working_time_ptr));
#endif
    
    GTAP_CUDA_TRY(gtap_finalize_runtime_error_report());

    return cudaGetLastError();
}

cudaError_t gtap_initialize(
    const gtap_block_config& config,
    size_t* device_bytes_allocated = nullptr
);

cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    gtap_block_config config;
    return gtap_initialize(config, device_bytes_allocated);
}

cudaError_t gtap_initialize(
    const gtap_block_config& config,
    size_t* device_bytes_allocated
) {
    cudaError_t validation = gtap_validate_config(config);
    if (validation != cudaSuccess) return validation;
    if (gtap_initialized_flag()) return cudaErrorInitializationError;
    gtap_launch_config launch_config{
        config.grid_size,
        gtap_compiled_block_size,
        (gtap_compiled_block_size + GTAP_WARP_SIZE - 1) / GTAP_WARP_SIZE,
        config.grid_size,
        config.max_tasks_per_block,
        1,
        config.max_tasks_per_block,
        config.profile_capacity_per_block,
        config.dynamic_shared_bytes
    };
    GTAP_CUDA_TRY(gtap_publish_launch_config(launch_config));
    gtap_stored_stream() = config.stream;
    cudaError_t err = __gtap_init_task_runtime();
    if (err == cudaSuccess) {
        gtap_initialized_flag() = true;
        gtap_store_optional_size(
            device_bytes_allocated, __gtap_runtime_device_allocation_bytes());
    }
    return err;
}

cudaError_t gtap_finalize() {
    cudaError_t err = __gtap_finalize_task_runtime();
    if (err == cudaSuccess) {
        gtap_initialized_flag() = false;
        gtap_stored_stream() = nullptr;
    }
    return err;
}

// Reset task runtime state for re-execution
// This function clears all runtime state without reallocating memory
// Call this before each execution after the initial init_task_runtime call
cudaError_t __gtap_reset_task_runtime() {
    gtap_reset_runtime_error_report_host();
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queues_ptr, d_block_task_queues, sizeof(BlockTaskQueue*)));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_block_task_queue_storage_ptr, d_block_task_queue_storage,
        sizeof(int*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    // Clear task queues
    if (d_block_task_queues_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_block_task_queues_ptr, 0, sizeof(BlockTaskQueue) * total_workers,
            streams[0]));
    }
    if (d_block_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_block_task_queue_storage_ptr, 0, sizeof(int) * total_tasks,
            streams[0]));
    }
    
    // Reset task ID pool metadata.
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_task_id_lists_ptr, 0, sizeof(TaskIdList) * total_workers,
            streams[1]));
    }
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks, streams[2]));
    }
    
    // Clear task data
    size_t max_task_size = gtap_host_task_data_stride();
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * total_tasks;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    }
    
    // Reset profile data if enabled
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    const size_t profile_bytes =
        sizeof(long long) * total_workers * gtap_profile_capacity();
    GTAP_CUDA_TRY(cudaMemsetAsync(
        having_task_time_ptr, 0, profile_bytes, streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_bytes, streams[1]));
#endif
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }
    
    // Reset global state
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    // Reinitialize block ID pools metadata
    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    
    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}

#ifdef GTAP_PROFILE
cudaError_t get_having_task_time_data(long long* host_having_task_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    return cudaMemcpy(
        host_having_task_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().grid_size *
            gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().grid_size *
            gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_having_task_time_data(int block_id, long long* host_having_task_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(
        host_having_task_time,
        ptr + static_cast<size_t>(block_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_block_working_time_data(int block_id, long long* host_working_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(
        host_working_time,
        ptr + static_cast<size_t>(block_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_final_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (having_task_time[blockIdx.x * gtap_profile_capacity() + i] > 0) {
                count++;
            }
        }
        indices[blockIdx.x] = count;
    }
}

__global__ void get_final_working_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (working_time[blockIdx.x * gtap_profile_capacity() + i] > 0) {
                count++;
            }
        }
        indices[blockIdx.x] = count;
    }
}
#endif

// Chase-Lev pop: owner pops from bottom
__device__ __forceinline__ int pop(int* taskId) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    
    int b = myQueue->bottom - 1;
    store_L2(&myQueue->bottom, b);
    __threadfence();
    
    int t = load_L2(&myQueue->top);
    int size = b - t;
    
    if (size < 0) {
        store_L2(&myQueue->bottom, t);
        *taskId = -1;
        return false;
    }
    
    int task_id = load_L2(gtap_block_queue_slot(
        blockIdx.x, b % d_gtap_launch_config.queue_capacity));
    
    if (size > 0) {
        *taskId = task_id;
#ifdef DEBUG
        printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
        return true;
    }
    
    if (atomicCAS(&myQueue->top, t, t + 1) != t) {
        *taskId = -1;
        store_L2(&myQueue->bottom, t + 1);
        return false;
    }
    
    *taskId = task_id;
    store_L2(&myQueue->bottom, t + 1);
#ifdef DEBUG
    printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
    return true;
}

template<TerminationMode M>
__device__ __forceinline__ int steal(int* taskId, bool prev_get_task) {
    int targetBlock = get_random_blocknum(blockIdx.x);
    BlockTaskQueue* targetBq = &d_block_task_queues[targetBlock];
    
    int t = load_L2(&targetBq->top);
    __threadfence();
    int b = load_L2(&targetBq->bottom);
    
    int size = b - t;
    if (size <= 0) {
        *taskId = -1;
        return false;
    }
    
    int task_id = load_L2(gtap_block_queue_slot(
        targetBlock, t % d_gtap_launch_config.queue_capacity));
    
    if (atomicCAS(&targetBq->top, t, t + 1) != t) {
        *taskId = -1;
        return false;
    }
    
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (!prev_get_task) atomicAdd(&d_active_worker_count, 1);
    }
    
    *taskId = task_id;
#ifdef DEBUG
    printf("steal: %d (block: %d -> %d)\n", task_id, targetBlock, blockIdx.x);
#endif
    return true;
}

// Chase-Lev push: owner pushes to bottom
__device__ __forceinline__ void push(
    TaskContext* ctx,
    int push_total,
    int* execute_task_id
) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    (void)push_total;

#ifdef GTAP_ASSUME_NO_TASKWAIT
    int publish_bottom = ctx->queue_tail;
    if (ctx->task_id_generated_count > 0) {
        publish_bottom = ctx->queue_tail - 1;
        if (threadIdx.x == 0) {
            *execute_task_id = load_L2(gtap_block_queue_slot(
                blockIdx.x,
                publish_bottom % d_gtap_launch_config.queue_capacity));
        }
    }
#else
    int publish_bottom = ctx->queue_tail;
    if (ctx->have_task_id_resumable) {
        if (threadIdx.x == 0) {
            *execute_task_id = ctx->task_id_resumable;
#ifdef DEBUG
            printf("resume: %d (block: %d)\n", *execute_task_id, blockIdx.x);
#endif
        }
    } else if (ctx->task_id_generated_count > 0) {
        publish_bottom = ctx->queue_tail - 1;
        if (threadIdx.x == 0) {
            *execute_task_id = load_L2(gtap_block_queue_slot(
                blockIdx.x,
                publish_bottom % d_gtap_launch_config.queue_capacity));
        }
    }
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_L2(&myQueue->bottom, publish_bottom);
    }
}


extern "C" __device__ __forceinline__ void __gtap_set_state_for_join(
    int tid,
    int child_count,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->state = next_state;
        hdr->waiting_child_count = child_count;
#endif
    }
}

extern "C" __device__ __forceinline__ bool __gtap_set_state_for_join_block(
    int tid,
    TaskContext* ctx,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    int child_count = ctx->task_id_generated_count;
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->state = next_state;
        hdr->waiting_child_count = child_count;
#endif
#ifdef DEBUG
    printf("set_state_for_join_block: tid=%d child_count=%d\n", tid, child_count);
#endif
    }
    __syncthreads();
    return child_count != 0;
}

__device__ __forceinline__ int __gtap_get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2_u16t(&d_task_headers[tid].state);
#endif
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
    if (rem == 1) {
        ctx->have_task_id_resumable = true;
        ctx->task_id_resumable = parentId;
    }
#ifdef DEBUG
    printf("notify_parent: %d, rem: %d\n", parentId, rem);
#endif
    return rem;
}
#endif

extern "C" __device__ void __gtap_finish_task(int tid, TaskContext* ctx) {
    __syncthreads();
    if (threadIdx.x == 0) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
        release_task_id_to_block_pool(tid);
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#else
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;
        
        if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifdef DEBUG
            printf("finish_task: %d, parent_tid: %d\n", tid, parent_tid);
#endif
            notify_parent(parent_tid, ctx);
            release_task_id_to_block_pool(tid);
        } else {
            release_task_id_to_block_pool(tid);
        }
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#endif
    }
}

extern "C" __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    int new_tid = get_task_id_from_block_pool(
        tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif
    
    reserve_unpublished_task_id(ctx, new_tid);
    (void)child_count;
    return __gtap_get_task_data(new_tid);
}

extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    TaskHeader* initial_hdr = &d_task_headers[0];
    initial_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)
    
    BlockTaskQueue* bq = &d_block_task_queues[blockIdx.x];
    store_L2(gtap_block_queue_slot(blockIdx.x, 0), 0);
    __threadfence();
    store_L2(&bq->bottom, 1);
}


template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    __shared__ int execute_task_id;
    __shared__ bool have_execute_task;
    __shared__ bool prev_get_task;
    __shared__ bool should_continue;
    __shared__ TaskContext block_ctx;
#ifdef GTAP_PROFILE
    __shared__ int having_task_time_idx;
    __shared__ int working_time_idx;
#endif

    if (threadIdx.x == 0) {
        should_continue = true;
        have_execute_task = false;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        block_ctx.have_task_id_resumable = false;
#endif
        block_ctx.task_id_generated_count = 0;
        block_ctx.id_list_free_pos_stale = d_gtap_launch_config.tasks_per_worker;
#ifdef GTAP_PROFILE
        working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            block_ctx.id_list_alloc_pos = 1;
            prev_get_task = true;
#ifdef GTAP_PROFILE
            having_task_time_idx = 1;
            having_task_time[blockIdx.x * gtap_profile_capacity()] =
                get_global_time();
#endif
        } else {
            block_ctx.id_list_alloc_pos = 0;
            prev_get_task = false;
#ifdef GTAP_PROFILE
            having_task_time_idx = 0;
#endif
        }
    }
    __syncthreads();
    
    while (should_continue) {
        if (threadIdx.x == 0) {
            if (!have_execute_task) {
                if (prev_get_task) {
                    have_execute_task = pop(&execute_task_id);
                }
            }
            if (!have_execute_task) {
                have_execute_task = steal<M>(&execute_task_id, prev_get_task);
            }
        }
        __syncthreads();

        if (!have_execute_task) {
            if (threadIdx.x == 0) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            bool all_tasks_finished = 1;
                            BlockTaskQueue* q = &d_block_task_queues[blockIdx.x];
                            int t = load_L2(&q->top);
                            int b = load_L2(&q->bottom);
                            if (t < b) {
                                all_tasks_finished = 0;
                            }
                            atomicExch(&d_all_tasks_finished_flag, all_tasks_finished);
                        }
                    }
                }
#ifdef GTAP_PROFILE
                if (prev_get_task && having_task_time_idx < gtap_profile_capacity()) {
                    having_task_time[
                        blockIdx.x * gtap_profile_capacity() +
                        having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = false;
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    should_continue = (load_L2(&d_all_tasks_finished_flag) == 0);
                    // if (active_worker_count == 0) consecutive_idle_count++;
                    // else consecutive_idle_count = 0;
                    // should_continue = (consecutive_idle_count != NUMBER_OF_CONSECUTIVE_IDLE_COUNTS_TO_TERMINATE);
                } else {
                    should_continue = (load_L2(&d_first_task_finished) == 0);
                }
            }
            __syncthreads();
            continue;
        } else {
            if (threadIdx.x == 0) {
#ifdef GTAP_PROFILE
                // Record task start time
                if (!prev_get_task && having_task_time_idx < gtap_profile_capacity()) {
                    having_task_time[
                        blockIdx.x * gtap_profile_capacity() +
                        having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = true;
                block_ctx.task_id_generated_count = 0;
                block_ctx.queue_tail = load_L2(&d_block_task_queues[blockIdx.x].bottom);
#ifndef GTAP_ASSUME_NO_TASKWAIT
                block_ctx.have_task_id_resumable = false;
#endif
            }
            __syncthreads();
        }

        if (have_execute_task) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            if (threadIdx.x == 0) {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &block_ctx.cached_task_header;
                dst_hdr->generation = load_L2_u16t(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2_u16t(&src_hdr->parent_generation);
            }
            __syncthreads();
#endif
            
#ifdef GTAP_PROFILE
            if (threadIdx.x == 0) {
                if (working_time_idx < gtap_profile_capacity()) {
                    working_time[
                        blockIdx.x * gtap_profile_capacity() +
                        working_time_idx] = get_global_time();
                    working_time_idx++;
                }
            }
#endif
            void* task_data = __gtap_get_task_data(execute_task_id);
            // Read function pointer atomically (64-bit) via L2 cache
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &block_ctx);
            // if(threadIdx.x == 0) printf("finish_execute_task: %d\n", tid);
        }
        __syncthreads();
        __threadfence();
#ifdef GTAP_PROFILE
        if (threadIdx.x == 0) {
            if (working_time_idx < gtap_profile_capacity()) {
                working_time[
                    blockIdx.x * gtap_profile_capacity() +
                    working_time_idx] = get_global_time();
                working_time_idx++;
            }
        }
#endif

        int total_count =
#ifdef GTAP_ASSUME_NO_TASKWAIT
            block_ctx.task_id_generated_count;
#else
            (block_ctx.have_task_id_resumable ? 1 : 0) + block_ctx.task_id_generated_count;
#endif
        int push_total = max(total_count - 1, 0);
        push(&block_ctx, push_total, &execute_task_id);
        if (threadIdx.x == 0) {
            // printf("total_count: %d\n", total_count);
            have_execute_task = (total_count > 0);
        }
    }
#ifdef DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

// Non-template device-side wrapper
extern "C" __device__ inline void __gtap_execute_task_loop_device() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}
