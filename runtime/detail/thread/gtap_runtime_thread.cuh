#pragma once

#include <cuda_runtime.h>
#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

#include "gtap_thread_core.cuh"

struct WarpTaskQueue {
    int count;
    int queue_lock;
    int queue_head;
    int queue_head_stale;
};

struct gtap_thread_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_warp = 150000;
    int num_queues = 1;
    int profile_capacity_per_warp = 30000;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_thread_config& config) {
    if (config.grid_size <= 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % GTAP_WARP_SIZE != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_warp <= 0 ||
        config.num_queues <= 0 ||
        config.max_tasks_per_warp % config.num_queues != 0) {
        return cudaErrorInvalidValue;
    }
#ifdef GTAP_PROFILE
    if (config.profile_capacity_per_warp <= 0) {
        return cudaErrorInvalidValue;
    }
#endif
    return cudaSuccess;
}

__constant__ WarpTaskQueue** d_warp_task_queues;
__constant__ int* d_warp_task_queue_storage;
extern __shared__ unsigned char __gtap_dynamic_shared[];

inline size_t gtap_thread_dynamic_shared_bytes(
    int block_size, int num_queues
) {
    const size_t warps = block_size / GTAP_WARP_SIZE;

    // Per-warp runtime context.
    size_t bytes = sizeof(TaskContext) * warps;
    // Align the following int arrays.
    bytes = gtap_align_up(bytes, alignof(int));
    // Per-warp, per-queue generated-task counters.
    bytes += sizeof(int) * warps * num_queues;
    // Per-warp, per-queue local queue tails.
    bytes += sizeof(int) * warps * num_queues;
    // Per-warp, per-queue staging slots, one slot per warp lane.
    bytes += sizeof(int) * warps * num_queues * GTAP_WARP_SIZE;
    // Per-warp, per-queue temporary counts used only by multi-queue DAQ.
    if (num_queues > 1) {
        bytes += sizeof(int) * warps * num_queues;
    }
#ifdef GTAP_PROFILE
    // Per-warp indices for having-task and working-time profile buffers.
    bytes += 2 * sizeof(int) * warps;
#endif
    return bytes;
}

__device__ __forceinline__ int* gtap_warp_queue_slot(
    int queue_idx, int warp_idx, int slot
) {
    const size_t index =
        (static_cast<size_t>(queue_idx) * d_gtap_launch_config.total_workers + warp_idx) * d_gtap_launch_config.queue_capacity + slot;
    return &d_warp_task_queue_storage[index];
}

static size_t __gtap_runtime_device_allocation_bytes() {
    const gtap_launch_config& c = gtap_stored_launch_config();
    const size_t workers = static_cast<size_t>(c.total_workers);
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t queue_ptr_array_bytes = sizeof(WarpTaskQueue*) * c.num_queues;
    const size_t queue_metadata_bytes =
        static_cast<size_t>(c.num_queues) * sizeof(WarpTaskQueue) * workers;
    const size_t queue_storage_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = gtap_host_task_data_stride() * tasks;
    const size_t task_id_metadata_bytes = sizeof(TaskIdList) * workers;
    const size_t task_id_storage_bytes = 2 * sizeof(int) * tasks;
    size_t total = queue_ptr_array_bytes + queue_metadata_bytes + queue_storage_bytes +
           header_bytes + task_data_bytes + task_id_metadata_bytes + task_id_storage_bytes;
#ifdef GTAP_PROFILE
    total += workers * gtap_profile_capacity() * (2 * sizeof(long long) + sizeof(int));
#endif
    return total;
}

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int queue_idx, int task_id) {
    int gen_idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[queue_idx], 1);
    if (gen_idx < GTAP_WARP_SIZE) {
        ctx->staged_task_ids[queue_idx * GTAP_WARP_SIZE + gen_idx] = task_id;
        return;
    }

    WarpTaskQueue* q = &d_warp_task_queues[queue_idx][get_warp_id_global()];
    int old_tail = atomicAdd(&ctx->tail_by_queue_idx[queue_idx], 1);
    int head = load_L2(&q->queue_head);
    const int queue_capacity = d_gtap_launch_config.queue_capacity;
    if (old_tail + 1 - head > queue_capacity - GTAP_QUEUE_MARGIN) {
        GTAP_RECORD_QUEUE_OVERFLOW(
            task_id, queue_idx, old_tail + 1 - head, queue_capacity - GTAP_QUEUE_MARGIN);
    }
    *gtap_warp_queue_slot(
        queue_idx, get_warp_id_global(), old_tail % queue_capacity) = task_id;
}

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    const int NUM_STREAMS = runtime_config.num_queues + 3;
    cudaStream_t* streams = reinterpret_cast<cudaStream_t*>(
        malloc(sizeof(cudaStream_t) * NUM_STREAMS));
    if (streams == nullptr) return cudaErrorMemoryAllocation;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    cudaEventRecord(start);
    #endif

    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queues_ptrptr), sizeof(WarpTaskQueue*) * runtime_config.num_queues));

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * runtime_config.num_queues, elapsed);
    #endif

    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * runtime_config.num_queues));
    for (int k = 0; k < runtime_config.num_queues; ++k) {
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(start);
        #endif
        WarpTaskQueue* plane_ptr = nullptr;
        GTAP_CUDA_TRY(cudaMalloc(
            reinterpret_cast<void**>(&plane_ptr),
            sizeof(WarpTaskQueue) * total_workers));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * total_workers, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_CUDA_TRY(cudaMemsetAsync(
            plane_ptr, 0, sizeof(WarpTaskQueue) * total_workers, streams[k]));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop, streams[k]);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemsetAsync(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * total_workers, elapsed);
        #endif
        h_warpTaskQueues_planes[k] = plane_ptr;
    }

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpy(d_warp_task_queues_ptrptr, h_warpTaskQueues_planes, sizeof(WarpTaskQueue*) * runtime_config.num_queues, cudaMemcpyHostToDevice));

    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_warp_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * runtime_config.num_queues, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * total_tasks));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n",
           sizeof(TaskHeader) * total_tasks, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks, streams[runtime_config.num_queues]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, streams[runtime_config.num_queues]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(TaskHeaders, %zu bytes): %.3f ms\n",
           sizeof(TaskHeader) * total_tasks, elapsed);
    cudaEventRecord(start);
    #endif

    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = gtap_host_task_data_stride();
    size_t task_data_size = max_task_size * total_tasks;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[runtime_config.num_queues + 1]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, streams[runtime_config.num_queues + 1]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_lists_ptr),
        sizeof(TaskIdList) * total_workers));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * total_workers, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * total_workers,
        streams[runtime_config.num_queues + 2]));
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr), sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_valid_ptr), sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_task_id_valid_ptr, 0, sizeof(int) * total_tasks,
        streams[runtime_config.num_queues + 2]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, streams[runtime_config.num_queues + 2]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * total_workers, elapsed);
    cudaEventRecord(start);
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queues, &d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue**)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &d_warp_task_queue_storage_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_warp_task_queues): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_headers): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_CUDA_TRY(gtap_init_device_task_data_stride());
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_data_bytes): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_lists): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    free(h_warpTaskQueues_planes);

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
    #endif
    // Initialize d_active_worker_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_worker_count): %.3f ms\n", elapsed);
    #endif

#ifdef GTAP_PROFILE
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    const size_t profile_long_bytes =
        sizeof(long long) * total_workers * gtap_profile_capacity();
    const size_t profile_int_bytes =
        sizeof(int) * total_workers * gtap_profile_capacity();
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&having_task_time_ptr), profile_long_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_long_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr), profile_int_bytes));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        having_task_time, &having_task_time_ptr, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        having_task_time_ptr, 0, profile_long_bytes, streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_long_bytes, streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0, profile_int_bytes,
        streams[2 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[2 % NUM_STREAMS]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size,
        runtime_config.warps_per_block * GTAP_WARP_SIZE>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  init_warp_id_pools_metadata kernel: %.3f ms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("=== init_task_runtime profiling complete ===\n\n");
    #endif

    return cudaGetLastError();
}

cudaError_t __gtap_finalize_task_runtime() {
    const int num_queues = gtap_stored_launch_config().num_queues;
    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
#endif

    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * num_queues));
    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * num_queues, cudaMemcpyDeviceToHost));

        // Free each queue plane
        for (int k = 0; k < num_queues; ++k) {
            if (h_warpTaskQueues_planes[k] != nullptr) {
                GTAP_CUDA_TRY(cudaFree(h_warpTaskQueues_planes[k]));
            }
        }
    }
    free(h_warpTaskQueues_planes);

    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_warp_task_queues_ptrptr));
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_warp_task_queue_storage_ptr));
    }

    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }

    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }

    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
    }
#ifdef GTAP_PROFILE
    if (having_task_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(having_task_time_ptr));
    if (working_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(working_time_ptr));
    if (tasks_processed_count_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(tasks_processed_count_ptr));
    }
#endif

    GTAP_CUDA_TRY(gtap_finalize_runtime_error_report());

    return cudaGetLastError();
}

cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated = nullptr
);

cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    gtap_thread_config config;
    return gtap_initialize(config, device_bytes_allocated);
}

cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated
) {
    cudaError_t validation = gtap_validate_config(config);
    if (validation != cudaSuccess) return validation;
    if (gtap_initialized_flag()) return cudaErrorInitializationError;

    gtap_launch_config launch_config{
        config.grid_size,
        config.block_size,
        config.block_size / GTAP_WARP_SIZE,
        config.grid_size * (config.block_size / GTAP_WARP_SIZE),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        config.profile_capacity_per_warp,
        gtap_thread_dynamic_shared_bytes(config.block_size, config.num_queues)
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

    const int NUM_STREAMS = runtime_config.num_queues + 3;
    cudaStream_t* streams = reinterpret_cast<cudaStream_t*>(
        malloc(sizeof(cudaStream_t) * NUM_STREAMS));
    if (streams == nullptr) return cudaErrorMemoryAllocation;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
#endif

    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * runtime_config.num_queues));
    GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * runtime_config.num_queues, cudaMemcpyDeviceToHost));

    // Clear task queues
    for (int k = 0; k < runtime_config.num_queues; ++k) {
        if (h_warpTaskQueues_planes[k] != nullptr) {
            GTAP_CUDA_TRY(cudaMemsetAsync(
                h_warpTaskQueues_planes[k], 0,
                sizeof(WarpTaskQueue) * total_workers, streams[k]));
        }
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    }
    free(h_warpTaskQueues_planes);

    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks,
            streams[runtime_config.num_queues]));
    }

    size_t max_task_size = gtap_host_task_data_stride();
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * total_tasks;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[runtime_config.num_queues + 1]));
    }

    // Reset task ID lists
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * total_workers,
            streams[runtime_config.num_queues + 2]));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_task_id_valid_ptr, 0, sizeof(int) * total_tasks,
            streams[runtime_config.num_queues + 2]));
    }

    // Reset global state
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));

    // Reset profile data if enabled
    #ifdef GTAP_PROFILE
    GTAP_CUDA_TRY(cudaMemsetAsync(
        having_task_time_ptr, 0,
        sizeof(long long) * total_workers * gtap_profile_capacity(), streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0,
        sizeof(long long) * total_workers * gtap_profile_capacity(),
        streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0,
        sizeof(int) * total_workers * gtap_profile_capacity(),
        streams[2 % NUM_STREAMS]));
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    // Reinitialize warp ID pools metadata
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size,
        runtime_config.warps_per_block * GTAP_WARP_SIZE>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}


#ifdef GTAP_PROFILE
cudaError_t get_warp_having_task_time_data(long long* host_having_task_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    return cudaMemcpy(
        host_having_task_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().total_workers *
            gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().total_workers *
            gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(int) * gtap_stored_launch_config().total_workers *
            gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_having_task_time_data(int warp_global_id, long long* host_having_task_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(
        host_having_task_time,
        ptr + static_cast<size_t>(warp_global_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_working_time_data(int warp_global_id, long long* host_working_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(
        host_working_time,
        ptr + static_cast<size_t>(warp_global_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_tasks_processed_count_data(int warp_global_id, int* host_counts, int max_samples) {
    int* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(
        host_counts,
        ptr + static_cast<size_t>(warp_global_id) * gtap_profile_capacity(),
        sizeof(int) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_final_warp_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (having_task_time[wid * gtap_profile_capacity() + i] > 0) count++;
        }
        indices[wid] = count;
    }
}

__global__ void get_final_warp_working_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (working_time[wid * gtap_profile_capacity() + i] > 0) count++;
        }
        indices[wid] = count;
    }
}
#endif

// define pop_batch, steal_batch, push_batch
__device__ __forceinline__ int pop_batch(int* execute_task_id, int max_count_to_pop, int* tail, int daq_idx) {
    int lane = get_lane_id();
    WarpTaskQueue* myQueue = &d_warp_task_queues[daq_idx][get_warp_id_global()];
    int pop_count = 0;
    if (lane == 0) {
        while (true) {
            int old_queue_count = load_L2(&myQueue->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_pop, old_queue_count);
            if (atomicCAS(&myQueue->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                pop_count = claim;
                *tail -= claim;
                break;
            }
        }
    }
    pop_count = __shfl_sync(0xFFFFFFFFu, pop_count, 0);
    if (lane >= GTAP_WARP_SIZE - max_count_to_pop && lane < GTAP_WARP_SIZE - max_count_to_pop + pop_count) {
        int pop_task_id = load_L2(gtap_warp_queue_slot(
            daq_idx,
            get_warp_id_global(),
            (*tail + (lane - GTAP_WARP_SIZE + max_count_to_pop)) %
                d_gtap_launch_config.queue_capacity));
#ifdef GTAP_INTERNAL_DEBUG
        printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", pop_task_id, daq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = pop_task_id;
    }
    return pop_count;
}

template<TerminationMode M>
__device__ __forceinline__ int steal_batch(int* execute_task_id, int max_count_to_steal, int daq_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    int old_head = 0;
    int steal_count = 0;
    WarpTaskQueue* targetWq = nullptr;
    if (lane == 0) {
        unsigned lock_backoff_ns = 32;
        while (true) {
            target_warp_id_global = get_random_warpnum_global(warp_id_global);
            targetWq = &d_warp_task_queues[daq_idx][target_warp_id_global];
            if (atomicCAS(&targetWq->queue_lock, 0, 1) == 0) break;
            __nanosleep(lock_backoff_ns);
            if (lock_backoff_ns < (1u << 12)) {
                lock_backoff_ns <<= 1u;
            }
        }
        while (true) {
            int old_queue_count = load_L2(&targetWq->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_steal, old_queue_count);
            if (atomicCAS(&targetWq->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (!prev_get_task) atomicAdd(&d_active_worker_count, 1);
                }
                steal_count = claim;
                old_head = load_L2(&targetWq->queue_head);
                break;
            }
        }
    }
    steal_count = __shfl_sync(0xFFFFFFFFu, steal_count, 0);
    if (steal_count == 0) {
        if (lane == 0) atomicExch(&targetWq->queue_lock, 0);
        return 0;
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    old_head = __shfl_sync(0xFFFFFFFFu, old_head, 0);
    if (lane >= GTAP_WARP_SIZE - max_count_to_steal && lane < GTAP_WARP_SIZE - max_count_to_steal + steal_count) {
        targetWq = &d_warp_task_queues[daq_idx][target_warp_id_global];
        int steal_task_id = load_L2(gtap_warp_queue_slot(
            daq_idx,
            target_warp_id_global,
            (old_head + (lane - GTAP_WARP_SIZE + max_count_to_steal)) %
                d_gtap_launch_config.queue_capacity));
#ifdef GTAP_INTERNAL_DEBUG
        printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", steal_task_id, daq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = steal_task_id;
    }
    __syncwarp();
    if (lane == 0) {
        targetWq->queue_head = old_head + steal_count;
        __threadfence();
        atomicExch(&targetWq->queue_lock, 0);
    }
    return steal_count;
}

__device__ __forceinline__ void push_batch (
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count,
    int* tail_by_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int k_max = 0;
    int max_gen = -1;
    int all_generated_count = 0;
    if (lane == 0) {
        for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
            int cnt = ctx->task_id_generated_count_by_queue_idx[k];
            all_generated_count += cnt;
            if (cnt > max_gen) {
                max_gen = cnt;
                k_max = k;
            }
        }
        ctx->queue_idx = k_max;
    }
    all_generated_count = __shfl_sync(0xFFFFFFFFu, all_generated_count, 0);
    if (all_generated_count == 0) {
        *execute_task_count = 0;
        return;
    }
    k_max = __shfl_sync(0xFFFFFFFFu, k_max, 0);
    max_gen = __shfl_sync(0xFFFFFFFFu, max_gen, 0);

    *execute_task_count = max(0, min(GTAP_WARP_SIZE, max_gen));
    if (lane < *execute_task_count) {
        *execute_task_id = ctx->staged_task_ids[k_max * GTAP_WARP_SIZE + lane];
#ifdef GTAP_INTERNAL_DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
    __syncwarp();

    for (int kind = 0; kind < d_gtap_launch_config.num_queues; ++kind) {
        int push_cnt = ctx->task_id_generated_count_by_queue_idx[kind];
        if (kind == k_max) {
            push_cnt -= *execute_task_count;
        }
        if (push_cnt <= 0) continue;

        WarpTaskQueue* q = &d_warp_task_queues[kind][warp_id_global];
        int total = ctx->task_id_generated_count_by_queue_idx[kind];
        int staged_n = min(total, GTAP_WARP_SIZE);
        if (kind != k_max) {
            for (int j = lane; j < staged_n; j += GTAP_WARP_SIZE) {
                *gtap_warp_queue_slot(
                    kind,
                    warp_id_global,
                    (tail_by_queue_idx[kind] + j) %
                        d_gtap_launch_config.queue_capacity) =
                    ctx->staged_task_ids[kind * GTAP_WARP_SIZE + j];
            }
            if (lane == 0) {
                tail_by_queue_idx[kind] += staged_n;
            }
            __syncwarp();
        }
        if (lane == 0) {
            atomicAdd(&q->count, push_cnt);
        }
    }
    if (lane == 0) {
        for (int kind = 0; kind < d_gtap_launch_config.num_queues; ++kind) {
            ctx->task_id_generated_count_by_queue_idx[kind] = 0;
        }
    }
}

// Get the current state of a task (reads from TaskHeader)
__device__ __forceinline__ int __gtap_get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2_u16t(&d_task_headers[tid].state);
#endif
}

__device__ __forceinline__ bool __gtap_set_state_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join >= d_gtap_launch_config.num_queues) {
        GTAP_RECORD_INVALID_QUEUE_IDX_AFTER_JOIN(
            tid, queue_idx_after_join, d_gtap_launch_config.num_queues);
    }
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* hdr = &d_task_headers[tid];
    hdr->queue_idx = queue_idx_after_join;
    hdr->state = next_state;
    hdr->waiting_child_count = child_count;
#else
    d_task_headers[tid].queue_idx = queue_idx_after_join;
    (void)next_state;
#endif
    return child_count != 0;
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef GTAP_INTERNAL_DEBUG
    int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parentId, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        int parent_queue_idx = load_L2_u16t(&parent_hdr->queue_idx);
        reserve_unpublished_task_id(ctx, parent_queue_idx, parentId);
    }
    return rem;
}
#endif

extern "C" __device__ __forceinline__ void __gtap_finish_task(int tid, TaskContext* ctx) {
#ifdef GTAP_INTERNAL_DEBUG
    printf("finish_task: %d in lane %d of warp %d of block %d\n", tid, get_lane_id(), get_warp_id_in_block(), blockIdx.x);
#endif

#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)ctx;
    release_task_id_to_warp_pool(tid);
#else
    int lane = get_lane_id();
    int parent_tid = ctx->task_parent_tids[lane];
    uint32_t cached_generations = ctx->task_generations[lane];
    uint16_t generation = static_cast<uint16_t>(cached_generations);
    uint16_t parent_generation =
        static_cast<uint16_t>(cached_generations >> 16);
    d_task_headers[tid].generation = generation + 1;

    if (tid != 0 &&
        load_L2_u16t(&d_task_headers[parent_tid].generation) ==
            parent_generation) {
        notify_parent(parent_tid, ctx);
    }
    release_task_id_to_warp_pool(tid);
#endif

    if (tid == 0) {
        store_L2(&d_first_task_finished, 1);
#ifdef GTAP_INTERNAL_DEBUG
        int lane = get_lane_id();
        printf("first task finished in lane %d of warp %d of block %d\n", lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
}

// Allocates task ID, sets up TaskHeader, returns task data pointer
// Caller stores task data fields after this call
extern "C" __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int child_queue_idx
) {
    if (child_queue_idx >= d_gtap_launch_config.num_queues) {
        GTAP_RECORD_INVALID_QUEUE_IDX(
            self_tid, child_queue_idx, d_gtap_launch_config.num_queues);
    }
    int warp_id_global = get_warp_id_global();
    int new_tid = get_task_id_from_warp_pool(
        &d_task_id_lists[warp_id_global],
        &ctx->id_list_alloc_pos,
        &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
    new_hdr->queue_idx = child_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int lane = get_lane_id();
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->task_generations[lane]);
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
#endif

    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);

#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
    return __gtap_get_task_data(new_tid);
}

extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int new_tid = 0;

    TaskHeader* initial_hdr = &d_task_headers[new_tid];
    initial_hdr->func = func;
    initial_hdr->queue_idx = initial_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)

    *gtap_warp_queue_slot(initial_queue_idx, warp_id_global, 0) = new_tid;
    __threadfence();
    // atomicExch(&d_active_worker_count, 1);
}

template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    const int warp_id_in_block = get_warp_id_in_block();
    const int warp_id_global = get_warp_id_global();
    const int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);
    bool should_continue = true;

    const int warps_per_block = d_gtap_launch_config.warps_per_block;
    TaskContext* warp_contexts =
        reinterpret_cast<TaskContext*>(__gtap_dynamic_shared);
    unsigned char* shared_cursor = __gtap_dynamic_shared +
        sizeof(TaskContext) * warps_per_block;
    shared_cursor = reinterpret_cast<unsigned char*>(
        gtap_align_up(
            reinterpret_cast<size_t>(shared_cursor), alignof(int)));
    const int num_queues = d_gtap_launch_config.num_queues;
    int* generated_counts = reinterpret_cast<int*>(shared_cursor);
    shared_cursor +=
        sizeof(int) * warps_per_block * num_queues;
    int* tail_by_queue_idx = reinterpret_cast<int*>(shared_cursor);
    shared_cursor +=
        sizeof(int) * warps_per_block * num_queues;
    int* staged_task_ids = reinterpret_cast<int*>(shared_cursor);
    shared_cursor += sizeof(int) * warps_per_block *
        num_queues * GTAP_WARP_SIZE;
    int* queue_counts = nullptr;
    if (num_queues > 1) {
        queue_counts = reinterpret_cast<int*>(shared_cursor);
        shared_cursor += sizeof(int) * warps_per_block * num_queues;
    }

    int* warp_generated_counts =
        generated_counts + warp_id_in_block * num_queues;
    int* warp_tails =
        tail_by_queue_idx + warp_id_in_block * num_queues;
    int* warp_staged =
        staged_task_ids + warp_id_in_block * num_queues * GTAP_WARP_SIZE;
    int* warp_queue_counts = num_queues > 1
        ? queue_counts + warp_id_in_block * num_queues
        : nullptr;

#ifdef GTAP_PROFILE
    int* having_time_idx = reinterpret_cast<int*>(shared_cursor);
    int* working_time_idx = having_time_idx + warps_per_block;
    if (lane == 0) {
        if (warp_id_global == 0) having_time_idx[warp_id_in_block] = 1;
        else having_time_idx[warp_id_in_block] = 0;
        working_time_idx[warp_id_in_block] = 0;
    }
#endif

    if (lane == 0) {
        warp_contexts[warp_id_in_block].queue_idx = 0;
        warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx =
            warp_generated_counts;
        warp_contexts[warp_id_in_block].tail_by_queue_idx = warp_tails;
        warp_contexts[warp_id_in_block].staged_task_ids = warp_staged;
        warp_contexts[warp_id_in_block].id_list_free_pos_stale =
            d_gtap_launch_config.tasks_per_worker;
        for (int k = 0; k < num_queues; ++k) {
            warp_generated_counts[k] = 0;
            warp_tails[k] = 0;
        }
        if (warp_id_global == 0) {
#ifdef GTAP_PROFILE
            having_task_time[warp_id_global * gtap_profile_capacity()] = get_global_time();
#endif
            warp_contexts[0].id_list_alloc_pos = 1;
            WarpTaskQueue* q = &d_warp_task_queues[0][0];
            store_L2(&q->count, 1);
            warp_tails[0] = 1;
        } else {
            warp_contexts[warp_id_in_block].id_list_alloc_pos = 0;
        }
    }
    __syncwarp();

    while (should_continue) {
        if (num_queues == 1) {
            // Single-queue fast path: skip DAQ count collection and selection.
            if (execute_task_count < GTAP_WARP_SIZE) {
                if (prev_get_task) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int pop_count = pop_batch(
                        &execute_task_id, remaining, &warp_tails[0], 0
                    );
                    execute_task_count += pop_count;
                }
                if (execute_task_count < GTAP_WARP_SIZE) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int steal_count = steal_batch<M>(
                        &execute_task_id, remaining, 0, prev_get_task
                    );
                    execute_task_count += steal_count;
                }
            }
        } else {
            // Multi-queue DAQ path.
            if (execute_task_count < GTAP_WARP_SIZE) {
                if (execute_task_count == 0) {
                    if (lane == 0) {
                        for (int k = 0; k < num_queues; ++k) {
                            warp_queue_counts[k] = load_L2(
                                &d_warp_task_queues[k][warp_id_global].count);
                        }
                    }
                    for (int attempt = 0; attempt < num_queues; ++attempt) {
                        int daq_idx;
                        if (lane == 0) {
                            daq_idx = gtap_select_next_fullest_queue_idx(
                                warp_queue_counts, num_queues);
                            warp_contexts[warp_id_in_block].queue_idx = daq_idx;
                        }
                        daq_idx = __shfl_sync(
                            0xFFFFFFFFu,
                            warp_contexts[warp_id_in_block].queue_idx,
                            0);
                        if (prev_get_task &&
                            execute_task_count < GTAP_WARP_SIZE) {
                            int remaining =
                                GTAP_WARP_SIZE - execute_task_count;
                            int pop_count = pop_batch(
                                &execute_task_id, remaining,
                                &warp_tails[daq_idx], daq_idx
                            );
                            execute_task_count += pop_count;
                        }
                        if (execute_task_count < GTAP_WARP_SIZE) {
                            int remaining =
                                GTAP_WARP_SIZE - execute_task_count;
                            int steal_count = steal_batch<M>(
                                &execute_task_id, remaining, daq_idx,
                                prev_get_task
                            );
                            execute_task_count += steal_count;
                        }
                        if (execute_task_count != 0) break;
                    }
                } else {
                    int daq_idx = __shfl_sync(
                        0xFFFFFFFFu,
                        warp_contexts[warp_id_in_block].queue_idx,
                        0
                    );
                    if (prev_get_task) {
                        int remaining = GTAP_WARP_SIZE - execute_task_count;
                        int pop_count = pop_batch(
                            &execute_task_id, remaining,
                            &warp_tails[daq_idx], daq_idx
                        );
                        execute_task_count += pop_count;
                    }
                    if (execute_task_count < GTAP_WARP_SIZE) {
                        int remaining = GTAP_WARP_SIZE - execute_task_count;
                        int steal_count = steal_batch<M>(
                            &execute_task_id, remaining, daq_idx, prev_get_task
                        );
                        execute_task_count += steal_count;
                    }
                }
            }
        }
        if (execute_task_count == 0) {
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            bool all_tasks_finished = 1;
                            for (int k = 0; k < num_queues; ++k) {
                                if (d_warp_task_queues[k][warp_id_global].queue_head < warp_tails[k]) {
                                    all_tasks_finished = 0;
                                    break;
                                }
                            }
                            atomicExch(&d_all_tasks_finished_flag, all_tasks_finished);
                        }
                    }
                }
                __syncwarp();
            }
#ifdef GTAP_PROFILE
            if (lane == 0) {
                if (prev_get_task && having_time_idx[warp_id_in_block] < gtap_profile_capacity()) {
                    having_task_time[
                        warp_id_global * gtap_profile_capacity() +
                        having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = false;
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) should_continue = (load_L2(&d_all_tasks_finished_flag) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            } else {
                if (lane == 0) should_continue = (load_L2(&d_first_task_finished) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            }
            continue;
        } else {
#ifdef GTAP_PROFILE
            if (lane == 0) {
                if (!prev_get_task && having_time_idx[warp_id_in_block] < gtap_profile_capacity()) {
                    having_task_time[
                        warp_id_global * gtap_profile_capacity() +
                        having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = true;
            if (lane == 0) {
                for (int k = 0; k < num_queues; ++k) {
                    warp_generated_counts[k] = 0;
                }
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            prefetch_global_L2(__gtap_get_task_data(execute_task_id));
            // Copy task header to TaskContext for reuse in task function (using L2 load)
#ifndef GTAP_ASSUME_NO_TASKWAIT
            {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskContext* dst_ctx = &warp_contexts[warp_id_in_block];
                uint16_t generation = load_L2_u16t(&src_hdr->generation);
                uint16_t parent_generation =
                    load_L2_u16t(&src_hdr->parent_generation);
                dst_ctx->task_parent_tids[lane] =
                    load_L2(&src_hdr->parent_tid);
                dst_ctx->task_generations[lane] =
                    static_cast<uint32_t>(generation) |
                    (static_cast<uint32_t>(parent_generation) << 16);
            }
#endif

#ifdef GTAP_PROFILE
            if (lane == 0) {
                if (working_time_idx[warp_id_in_block] < gtap_profile_capacity()) {
                    const int profile_idx =
                        warp_id_global * gtap_profile_capacity() +
                        working_time_idx[warp_id_in_block];
                    working_time[profile_idx] = get_global_time();
                    tasks_processed_count[profile_idx] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                }
            }
#endif
            void* task_data = __gtap_get_task_data(execute_task_id);
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &warp_contexts[warp_id_in_block]);
#ifdef GTAP_INTERNAL_DEBUG
            printf("executed_task_id: %d in lane %d of warp %d of block %d\n", execute_task_id, lane, warp_id_in_block, blockIdx.x);
#endif
            
        }
        __syncwarp();
        __threadfence();
#ifdef GTAP_PROFILE
        if (lane == 0) {
            if (working_time_idx[warp_id_in_block] < gtap_profile_capacity()) {
                const int profile_idx =
                    warp_id_global * gtap_profile_capacity() +
                    working_time_idx[warp_id_in_block];
                working_time[profile_idx] = get_global_time();
                tasks_processed_count[profile_idx] = execute_task_count;
                working_time_idx[warp_id_in_block]++;
            }
        }
        __syncwarp();
#endif

        push_batch(
            &warp_contexts[warp_id_in_block], &execute_task_id,
            &execute_task_count, warp_tails
        );
    }
#ifdef GTAP_INTERNAL_DEBUG
    if (lane == 0) printf("execute_task_loop: end (warp_id_global = %d)\n", warp_id_global);
#endif
}

// Non-template device-side wrapper
extern "C" __device__ __forceinline__ void __gtap_execute_task_loop_device() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}
