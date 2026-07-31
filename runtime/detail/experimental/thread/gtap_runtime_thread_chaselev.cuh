#pragma once

#include <cuda_runtime.h>
#include "../../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

#define GTAP_THREAD_HAS_GENERATED_TASK_IDS 1

#include "../../thread/gtap_thread_core.cuh"

struct gtap_thread_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_warp = 150000;
    int num_queues = 1;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_thread_config& config) {
    if (config.grid_size <= 0 || config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % GTAP_WARP_SIZE != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_warp <= 0 || config.num_queues <= 0 ||
        config.max_tasks_per_warp % config.num_queues != 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

struct WarpTaskQueue {
    int top;           // Chase-Lev top (steal from here)
    int bottom;        // Chase-Lev bottom (push/pop here)
};

__constant__ WarpTaskQueue** d_warp_task_queues;
__constant__ int* d_warp_task_queue_storage;
extern __shared__ unsigned char __gtap_dynamic_shared[];

inline size_t gtap_thread_dynamic_shared_bytes(
    int block_size, int num_queues
) {
    const size_t warps = block_size / GTAP_WARP_SIZE;
    size_t bytes = sizeof(TaskContext) * warps;
    bytes = gtap_align_up(bytes, alignof(int));
    bytes += sizeof(int) * warps * num_queues;
    if (num_queues > 1) {
        bytes += sizeof(int) * warps * num_queues;
    }
#ifdef PROFILE
    bytes += 2 * sizeof(int) * warps;
#endif
    return bytes;
}

__device__ __forceinline__ int* gtap_chaselev_queue_slot(
    int queue_idx, int worker_idx, int slot
) {
    const size_t index =
        (static_cast<size_t>(queue_idx) *
             d_gtap_launch_config.total_workers +
         worker_idx) *
            d_gtap_launch_config.queue_capacity +
        slot;
    return &d_warp_task_queue_storage[index];
}

#define GTAP_RUNTIME_GRID_SIZE (gtap_stored_launch_config().grid_size)
#define GTAP_RUNTIME_BLOCK_SIZE (gtap_stored_launch_config().block_size)
#define GTAP_RUNTIME_NUM_WARPS (gtap_stored_launch_config().warps_per_block)
#define GTAP_RUNTIME_TOTAL_WORKERS (gtap_stored_launch_config().total_workers)
#define GTAP_RUNTIME_TASKS_PER_WORKER \
    (gtap_stored_launch_config().tasks_per_worker)
#define GTAP_RUNTIME_NUM_QUEUES (gtap_stored_launch_config().num_queues)
#define GTAP_RUNTIME_QUEUE_CAPACITY \
    (gtap_stored_launch_config().queue_capacity)
#define GTAP_RUNTIME_TOTAL_TASKS \
    (GTAP_RUNTIME_TOTAL_WORKERS * GTAP_RUNTIME_TASKS_PER_WORKER)

static size_t __gtap_runtime_device_allocation_bytes() {
    const size_t queue_ptr_array_bytes = sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES;
    const size_t queue_plane_bytes =
        (size_t)GTAP_RUNTIME_NUM_QUEUES * sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS;
    const size_t header_bytes = sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_data_bytes = gtap_host_task_data_stride() * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_id_list_bytes = sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS;
    const size_t task_id_generated_bytes = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS *
                                           GTAP_RUNTIME_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS + 1) *
                                           GTAP_WARP_SIZE;
    const size_t queue_storage_bytes =
        sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_id_pool_bytes =
        2 * sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS;
    size_t total =
        queue_ptr_array_bytes + queue_plane_bytes + header_bytes + task_data_bytes +
           task_id_list_bytes + task_id_generated_bytes +
           queue_storage_bytes + task_id_pool_bytes;
#ifdef PROFILE
    total += GTAP_RUNTIME_TOTAL_WORKERS * MAX_PROFILE_DATA *
        (2 * sizeof(long long) + sizeof(int));
#endif
    return total;
}

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_tasks =
        static_cast<size_t>(runtime_config.total_workers) *
        runtime_config.tasks_per_worker;

    #ifdef INIT_PROFILE
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    
    cudaEventRecord(start);
    #endif

    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queues_ptrptr), sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES));
    
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES, elapsed);
    #endif
    
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES));
    for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
        #ifdef INIT_PROFILE
        cudaEventRecord(start);
        #endif
        WarpTaskQueue* plane_ptr = nullptr;
        GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&plane_ptr), sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_CUDA_TRY(cudaMemset(plane_ptr, 0, sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemset(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
        #endif
        h_warpTaskQueues_planes[k] = plane_ptr;
    }
    
    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpy(d_warp_task_queues_ptrptr, h_warpTaskQueues_planes, sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyHostToDevice));

    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_warp_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemset(
        d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, elapsed);
    cudaEventRecord(start);
    #endif

    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = gtap_host_task_data_stride();
    size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);  
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemset(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif

    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS + 1) * GTAP_WARP_SIZE;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_by_queue_idx_ptr), task_id_array_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(task_id_generated, %zu bytes): %.3f ms\n", task_id_array_size, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(task_id_generated, %zu bytes): %.3f ms\n", task_id_array_size, elapsed);
    cudaEventRecord(start);
    #endif


    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queues, &d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue**)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &d_warp_task_queue_storage_ptr,
        sizeof(int*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_warp_task_queues): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_headers): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_CUDA_TRY(gtap_init_device_task_data_stride());
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_data_bytes): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_lists): %.3f ms\n", elapsed); 
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx, &d_task_id_generated_by_queue_idx_ptr, sizeof(int*)));
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_valid_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemset(
        d_task_id_valid_ptr, 0, sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx): %.3f ms\n", elapsed);
    #endif

    free(h_warpTaskQueues_planes);

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
    #endif
    // Initialize d_active_worker_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_worker_count): %.3f ms\n", elapsed);
    #endif

#ifdef PROFILE
    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    long long (*having_task_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    long long (*working_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    int (*tasks_processed_count_ptr)[MAX_PROFILE_DATA] = nullptr;
    const size_t profile_time_bytes =
        sizeof(long long) * runtime_config.total_workers * MAX_PROFILE_DATA;
    const size_t profile_count_bytes =
        sizeof(int) * runtime_config.total_workers * MAX_PROFILE_DATA;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&having_task_time_ptr),
        profile_time_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr),
        profile_time_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr),
        profile_count_bytes));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        having_task_time, &having_task_time_ptr,
        sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr,
        sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_CUDA_TRY(cudaMemset(
        having_task_time_ptr, 0, profile_time_bytes));
    GTAP_CUDA_TRY(cudaMemset(
        working_time_ptr, 0, profile_time_bytes));
    GTAP_CUDA_TRY(cudaMemset(
        tasks_processed_count_ptr, 0, profile_count_bytes));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
#endif

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0,
        gtap_stored_stream()>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    #ifdef INIT_PROFILE
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
    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage,
        sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));
#ifdef PROFILE
    long long (*having_task_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    long long (*working_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    int (*tasks_processed_count_ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time,
        sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time,
        sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
#endif

    
    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES));
    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyDeviceToHost));
        
        // Free each queue plane
        for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
            if (h_warpTaskQueues_planes[k] != nullptr) {
                GTAP_CUDA_TRY(cudaFree(h_warpTaskQueues_planes[k]));
            }
        }
    }
    free(h_warpTaskQueues_planes);
    
    // Free queue pointer array
    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_warp_task_queues_ptrptr));
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_warp_task_queue_storage_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
    }
    
    // Free other allocated memory
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_generated_by_queue_idx_ptr));
    }
#ifdef PROFILE
    if (having_task_time_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(having_task_time_ptr));
    }
    if (working_time_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(working_time_ptr));
    }
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
    gtap_launch_config runtime_config{
        config.grid_size,
        config.block_size,
        config.block_size / GTAP_WARP_SIZE,
        config.grid_size * (config.block_size / GTAP_WARP_SIZE),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        gtap_thread_dynamic_shared_bytes(
            config.block_size, config.num_queues)
    };
    GTAP_CUDA_TRY(gtap_publish_launch_config(runtime_config));
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
    const gtap_launch_config& runtime_config =
        gtap_stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks =
        total_workers * runtime_config.tasks_per_worker;

    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage,
        sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));
#ifdef PROFILE
    long long (*having_task_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    long long (*working_time_ptr)[MAX_PROFILE_DATA] = nullptr;
    int (*tasks_processed_count_ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time,
        sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time,
        sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
#endif

    
    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES));
    GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyDeviceToHost));
    
    // Clear task queues
    for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
        if (h_warpTaskQueues_planes[k] != nullptr) {
            GTAP_CUDA_TRY(cudaMemset(h_warpTaskQueues_planes[k], 0, sizeof(WarpTaskQueue) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        }
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemset(
            d_warp_task_queue_storage_ptr, 0,
            sizeof(int) * total_tasks));
    }
    free(h_warpTaskQueues_planes);
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    }
    
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t max_task_size = gtap_host_task_data_stride();
        size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
        GTAP_CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    }
    
    // Reset task ID lists
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemset(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemset(
            d_task_id_valid_ptr, 0, sizeof(int) * total_tasks));
    }
    
    // Clear task ID generated array
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS + 1) * GTAP_WARP_SIZE;
        GTAP_CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    }

    // Reset global state
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));

    // Reset profile data if enabled
    #ifdef PROFILE
    GTAP_CUDA_TRY(cudaMemset(
        having_task_time_ptr, 0,
        sizeof(long long) * total_workers * MAX_PROFILE_DATA));
    GTAP_CUDA_TRY(cudaMemset(
        working_time_ptr, 0,
        sizeof(long long) * total_workers * MAX_PROFILE_DATA));
    GTAP_CUDA_TRY(cudaMemset(
        tasks_processed_count_ptr, 0,
        sizeof(int) * total_workers * MAX_PROFILE_DATA));
    #endif
    
    // Reinitialize warp ID pools metadata
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0,
        gtap_stored_stream()>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    
    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}


#ifdef PROFILE
cudaError_t get_warp_having_task_time_data(long long* host_having_task_time) {
    long long (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    return cudaMemcpy(
        host_having_task_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().total_workers *
            MAX_PROFILE_DATA,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(
        host_working_time, ptr,
        sizeof(long long) * gtap_stored_launch_config().total_workers *
            MAX_PROFILE_DATA,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(
        host_counts, ptr,
        sizeof(int) * gtap_stored_launch_config().total_workers *
            MAX_PROFILE_DATA,
        cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_having_task_time_data(int warp_global_id, long long* host_having_task_time, int max_samples) {
    long long (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    const int count =
        max_samples < MAX_PROFILE_DATA ? max_samples : MAX_PROFILE_DATA;
    return cudaMemcpy(
        host_having_task_time, ptr[warp_global_id],
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_working_time_data(int warp_global_id, long long* host_working_time, int max_samples) {
    long long (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count =
        max_samples < MAX_PROFILE_DATA ? max_samples : MAX_PROFILE_DATA;
    return cudaMemcpy(
        host_working_time, ptr[warp_global_id],
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_tasks_processed_count_data(int warp_global_id, int* host_counts, int max_samples) {
    int (*ptr)[MAX_PROFILE_DATA] = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &ptr, tasks_processed_count, sizeof(ptr)));
    const int count =
        max_samples < MAX_PROFILE_DATA ? max_samples : MAX_PROFILE_DATA;
    return cudaMemcpy(
        host_counts, ptr[warp_global_id],
        sizeof(int) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_final_warp_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (having_task_time[wid][i] > 0) count++;
        }
        indices[wid] = count;
    }
}

__global__ void get_final_warp_working_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (working_time[wid][i] > 0) count++;
        }
        indices[wid] = count;
    }
}
#endif

// Chase-Lev style sequential pop/steal operations

// Chase-Lev popBottom (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty)
__device__ __forceinline__ int pop_single_chase_lev(
    WarpTaskQueue* q, int queue_idx, int worker_idx
) {
    int b = q->bottom - 1;
    store_L2(&q->bottom, b);
    __threadfence();
    int t = load_L2(&q->top);
    int size = b - t;
    
    if (size < 0) {
        q->bottom = t;
        return -1;
    }
    
    int task_id = load_L2(gtap_chaselev_queue_slot(
        queue_idx, worker_idx,
        b % d_gtap_launch_config.queue_capacity));
    
    if (size > 0) {
        return task_id;
    }
    
    if (atomicCAS(&q->top, t, t + 1) != t) {
        // Lost race to stealer
        task_id = -1;
    }
    store_L2(&q->bottom, t + 1);
    return task_id;
}

// Sequential pop using chase-lev (repeats single pops)
__device__ __forceinline__ int pop_chase_lev(int* execute_task_id, int max_count_to_pop, int daq_idx) {
    int lane = get_lane_id();
    WarpTaskQueue* myQueue = &d_warp_task_queues[daq_idx][get_warp_id_global()];
    int pop_count = 0;
    
    for (int i = 0; i < max_count_to_pop; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = pop_single_chase_lev(
                myQueue, daq_idx, get_warp_id_global());
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);
        
        if (task_id == -1) break;
        
        // Assign to lane (filling from high lanes: GTAP_WARP_SIZE-max_count_to_pop, ...)
        int target_lane = GTAP_WARP_SIZE - max_count_to_pop + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef DEBUG
            printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, daq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        pop_count++;
    }
    
    return pop_count;
}

// Chase-Lev steal (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty or Abort)
__device__ __forceinline__ int steal_single_chase_lev(
    WarpTaskQueue* q, int queue_idx, int worker_idx
) {
    int t = load_L2(&q->top);
    __threadfence();
    int b = load_L2(&q->bottom);
    
    int size = b - t;
    if (size <= 0) return -1;
    
    int task_id = load_L2(gtap_chaselev_queue_slot(
        queue_idx, worker_idx,
        t % d_gtap_launch_config.queue_capacity));
    
    if (atomicCAS(&q->top, t, t + 1) != t) {
        return -1;  // Abort - lost race
    }
    
    return task_id;
}

// Sequential steal using chase-lev (repeats single steals)
template<TerminationMode M>
__device__ __forceinline__ int steal_chase_lev(int* execute_task_id, int max_count_to_steal, int daq_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    WarpTaskQueue* targetWq = nullptr;
    int steal_count = 0;
    bool active_count_incremented = false;
    
    // Select a random victim (lane 0 only)
    if (lane == 0) {
        target_warp_id_global = get_random_warpnum_global(warp_id_global);
        targetWq = &d_warp_task_queues[daq_idx][target_warp_id_global];
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    targetWq = &d_warp_task_queues[daq_idx][target_warp_id_global];
    
    // Sequential steals using chase-lev
    for (int i = 0; i < max_count_to_steal; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = steal_single_chase_lev(
                targetWq, daq_idx, target_warp_id_global);
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);
        
        if (task_id == -1) break;
        
        // Increment active worker count on first successful steal
        if (M == TERMINATE_ON_ALL_TASKS_FINISH && !active_count_incremented && !prev_get_task) {
            if (lane == 0) atomicAdd(&d_active_worker_count, 1);
            active_count_incremented = true;
        }
        
        // Assign to lane (filling from high lanes: GTAP_WARP_SIZE-max_count_to_steal, ...)
        int target_lane = GTAP_WARP_SIZE - max_count_to_steal + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef DEBUG
            printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, daq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        steal_count++;
    }
    
    return steal_count;
}

// Chase-Lev pushBottom (multiple items)
// NOTE: the template parameter is not used
template<TerminationMode M>
__device__ __forceinline__ void push_batch (
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
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
        *execute_task_id = get_task_id_generated(warp_id_global, k_max, lane);
#ifdef DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }

    #pragma unroll
    for (int kind = 0; kind < d_gtap_launch_config.num_queues; ++kind) {
        int first_idx_to_push = (kind == k_max) ? *execute_task_count : 0;
        int push_cnt = ctx->task_id_generated_count_by_queue_idx[kind] - first_idx_to_push;
        if (push_cnt <= 0) continue;

        WarpTaskQueue* q = &d_warp_task_queues[kind][warp_id_global];
        
        int b = load_L2(&q->bottom);
        int t = load_L2(&q->top);
        
        if (lane == 0) {
            int size = b - t;
            if (size + push_cnt >= d_gtap_launch_config.queue_capacity - GTAP_QUEUE_MARGIN) {
                GTAP_RECORD_QUEUE_OVERFLOW(
                    -1, kind, size + push_cnt, d_gtap_launch_config.queue_capacity - GTAP_QUEUE_MARGIN);
            }
        }

        for (int j = lane; j < push_cnt; j += GTAP_WARP_SIZE) {
            int idx_to_push = (b + j) % d_gtap_launch_config.queue_capacity;
            int val = get_task_id_generated(warp_id_global, kind, first_idx_to_push + j);
            *gtap_chaselev_queue_slot(
                kind, warp_id_global, idx_to_push) = val;
#ifdef DEBUG
            printf("push_task_id: %d to %d (kind %d) in lane %d of warp %d of block %d\n", val, idx_to_push, kind, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        __threadfence();
        __syncwarp();
        
        if (lane == 0) {
            store_L2(&q->bottom, b + push_cnt);
        }
    }
}

extern "C" {
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
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef DEBUG
    int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parentId, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        int parent_queue_idx = load_L2_u16t(&parent_hdr->queue_idx);
        int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[parent_queue_idx], 1);
        set_task_id_generated(get_warp_id_global(), parent_queue_idx, idx, parentId);
    }
    return rem;
}
#endif

extern "C" __device__ __forceinline__ void __gtap_finish_task(int tid, TaskContext* ctx) {
#ifdef DEBUG
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
#ifdef DEBUG
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
        GTAP_RECORD_INVALID_QUEUE_IDX(self_tid, child_queue_idx, d_gtap_launch_config.num_queues);
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
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->task_generations[lane]);
    new_hdr->waiting_child_count = 0;
#endif
    
    int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[child_queue_idx], 1);
    set_task_id_generated(warp_id_global, child_queue_idx, idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
    return __gtap_get_task_data(new_tid);
}


// Non-template version that takes a pointer and size for compiler-generated code
extern "C" __device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size,
    int child_queue_idx
) {
    if (child_queue_idx >= d_gtap_launch_config.num_queues) {
        GTAP_RECORD_INVALID_QUEUE_IDX(self_tid, child_queue_idx, d_gtap_launch_config.num_queues);
    }

    int warp_id_global = get_warp_id_global();
    TaskIdList* tid_list = &d_task_id_lists[warp_id_global];
    int new_tid = get_task_id_from_warp_pool(
        tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
    new_hdr->queue_idx = child_queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int lane = get_lane_id();
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation =
        static_cast<uint16_t>(ctx->task_generations[lane]);
    new_hdr->waiting_child_count = 0;
#endif

    // Copy task data atomically word-by-word
    void* dest_task = __gtap_get_task_data(new_tid);
    memcpy(dest_task, task_data_ptr, task_data_size);
    // __gtap_copy_bytes(dest_task, task_data_ptr, task_data_size);
    
    int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[child_queue_idx], 1);
    set_task_id_generated(warp_id_global, child_queue_idx, idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
}

// Non-template version for compiler-generated code using raw pointer and size
// __gtap_push_initial_task: Device function to push initial task
// This function is called from compiler-generated kernel code
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

    *gtap_chaselev_queue_slot(
        initial_queue_idx, warp_id_global, 0) = new_tid;
    __threadfence();
    // atomicExch(&d_active_worker_count, 1);
}

}  // extern "C"

template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    int warp_id_in_block = get_warp_id_in_block();
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);
    bool should_continue = true;

    TaskContext* warp_contexts =
        reinterpret_cast<TaskContext*>(__gtap_dynamic_shared);
    unsigned char* shared_cursor = __gtap_dynamic_shared +
        sizeof(TaskContext) * d_gtap_launch_config.warps_per_block;
    shared_cursor = reinterpret_cast<unsigned char*>(gtap_align_up(
        reinterpret_cast<size_t>(shared_cursor), alignof(int)));
    int* generated_counts = reinterpret_cast<int*>(shared_cursor);
    shared_cursor += sizeof(int) *
        d_gtap_launch_config.warps_per_block *
        d_gtap_launch_config.num_queues;
    int* queue_counts = d_gtap_launch_config.num_queues > 1
        ? reinterpret_cast<int*>(shared_cursor)
        : nullptr;

#ifdef PROFILE
    if (d_gtap_launch_config.num_queues > 1) {
        shared_cursor += sizeof(int) *
            d_gtap_launch_config.warps_per_block *
            d_gtap_launch_config.num_queues;
    }
    int* having_time_idx = reinterpret_cast<int*>(shared_cursor);
    int* working_time_idx =
        having_time_idx + d_gtap_launch_config.warps_per_block;
    if (lane == 0) {
        if (warp_id_global == 0) having_time_idx[warp_id_in_block] = 1;
        else having_time_idx[warp_id_in_block] = 0;
        working_time_idx[warp_id_in_block] = 0;
    }
#endif

    if (lane == 0) {
        warp_contexts[warp_id_in_block].
            task_id_generated_count_by_queue_idx =
                generated_counts +
                warp_id_in_block * d_gtap_launch_config.num_queues;
        warp_contexts[warp_id_in_block].queue_idx = 0;
        warp_contexts[warp_id_in_block].id_list_free_pos_stale = d_gtap_launch_config.tasks_per_worker;
        #pragma unroll
        for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
            warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
        }
        if (warp_id_global == 0) {
#ifdef PROFILE
            having_task_time[warp_id_global][0] = get_global_time();
#endif
            warp_contexts[0].id_list_alloc_pos = 1;
            // Chase-Lev: set bottom = 1 (initial task at position 0)
            WarpTaskQueue* q = &d_warp_task_queues[0][0];
            q->bottom = 1;
        } else {
            warp_contexts[warp_id_in_block].id_list_alloc_pos = 0;
        }
    }
    __syncwarp();
    
    while (should_continue) {
        if (execute_task_count == 0) {
            if (d_gtap_launch_config.num_queues > 1) {
            int* warp_queue_counts = queue_counts +
                warp_id_in_block * d_gtap_launch_config.num_queues;
            if (lane == 0) {
                for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
                    WarpTaskQueue* q = &d_warp_task_queues[k][warp_id_global];
                    warp_queue_counts[k] =
                        load_L2(&q->bottom) - load_L2(&q->top);
                }
            }
            for (int attempt = 0; attempt < d_gtap_launch_config.num_queues; ++attempt) {
                int daq_idx;
                if (lane == 0) {
                    daq_idx = gtap_select_next_fullest_queue_idx(
                        warp_queue_counts,
                        d_gtap_launch_config.num_queues);
                    warp_contexts[warp_id_in_block].queue_idx = daq_idx;
                }
                daq_idx = __shfl_sync(0xFFFFFFFFu, warp_contexts[warp_id_in_block].queue_idx, 0);
                if (prev_get_task && execute_task_count < GTAP_WARP_SIZE) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int pop_count = pop_chase_lev(&execute_task_id, remaining, daq_idx);
                    execute_task_count += pop_count;
                }
                if (execute_task_count < GTAP_WARP_SIZE) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int steal_count = steal_chase_lev<M>(&execute_task_id, remaining, daq_idx, prev_get_task);
                    execute_task_count += steal_count;
                }
                if (execute_task_count != 0) break;
            }
            } else {
            if (prev_get_task && execute_task_count < GTAP_WARP_SIZE) {
                int remaining = GTAP_WARP_SIZE - execute_task_count;
                int pop_count = pop_chase_lev(&execute_task_id, remaining, 0);
                execute_task_count += pop_count;
            }
            if (execute_task_count < GTAP_WARP_SIZE) {
                int remaining = GTAP_WARP_SIZE - execute_task_count;
                int steal_count = steal_chase_lev<M>(&execute_task_id, remaining, 0, prev_get_task);
                execute_task_count += steal_count;
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
                            #pragma unroll
                            for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
                                // Chase-Lev: check if queue is empty (top >= bottom)
                                WarpTaskQueue* q = &d_warp_task_queues[k][warp_id_global];
                                if (q->top < q->bottom) {
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
#ifdef PROFILE
            if (lane == 0) {
                if (prev_get_task && having_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    having_task_time[warp_id_global][having_time_idx[warp_id_in_block]] = get_global_time();
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
#ifdef PROFILE
            if (lane == 0) {
                if (!prev_get_task && having_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    having_task_time[warp_id_global][having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = true;
            if (lane == 0) {
                for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            prefetch_global_L2(__gtap_get_task_data(execute_task_id));
            // unsigned active_mask = __activemask();
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
            // __syncwarp(active_mask);
            
#ifdef PROFILE
            if (lane == 0) {
                if (working_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    working_time[warp_id_global][working_time_idx[warp_id_in_block]] = get_global_time();
                    tasks_processed_count[warp_id_global][working_time_idx[warp_id_in_block]] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                }
            }
#endif
            // Use non-template version to avoid TaskType dependency
            void* task_data = __gtap_get_task_data(execute_task_id);
            // printf("task_data: %p\n", task_data);
            // if (lane == 0) {
            //     printf("execute_task_loop: execute_task_id = %d, d_task_headers[%d].func = %p\n", execute_task_id, execute_task_id, d_task_headers[execute_task_id].func);
            // }
            // Read function pointer atomically (64-bit)
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &warp_contexts[warp_id_in_block]);
#ifdef DEBUG
            printf("executed_task_id: %d in lane %d of warp %d of block %d\n", execute_task_id, lane, warp_id_in_block, blockIdx.x);
#endif
            __threadfence();
        }
        __syncwarp();
#ifdef PROFILE
        if (lane == 0) {
            if (working_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                working_time[warp_id_global][working_time_idx[warp_id_in_block]] = get_global_time();
                tasks_processed_count[warp_id_global][working_time_idx[warp_id_in_block]] = execute_task_count;
                working_time_idx[warp_id_in_block]++;
            }
        }
        __syncwarp();
#endif

        push_batch<M>(
            &warp_contexts[warp_id_in_block], &execute_task_id,
            &execute_task_count
        );
    }
#ifdef DEBUG
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
