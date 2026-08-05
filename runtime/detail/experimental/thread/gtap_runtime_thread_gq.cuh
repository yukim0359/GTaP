#pragma once

#include <cuda_runtime.h>
#include "../../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

#include "../../thread/gtap_thread_core.cuh"

constexpr int GTAP_TASK_ID_GEN_QUEUE_STRIDE =
    GTAP_MAX_CHILD_TASKS * GTAP_WARP_SIZE;

struct gtap_thread_config {
    int grid_size = 1024;
    int block_size = 256;
    int max_tasks_per_warp = 20000;
    int num_queues = 1;
    int profile_capacity_per_warp = 30000;
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
#ifdef GTAP_PROFILE
    if (config.profile_capacity_per_warp <= 0) return cudaErrorInvalidValue;
#endif
    return cudaSuccess;
}

__constant__ int* d_global_task_queue;
__constant__ int* d_queue_head;
__constant__ int* d_queue_tail;
__constant__ int* d_queue_alloc;
__constant__ int* d_task_id_generated_by_queue_idx;
extern __shared__ unsigned char __gtap_dynamic_shared[];

#ifdef __CUDA_ARCH__
#define GTAP_RUNTIME_GRID_SIZE (d_gtap_launch_config.grid_size)
#define GTAP_RUNTIME_BLOCK_SIZE (d_gtap_launch_config.block_size)
#define GTAP_RUNTIME_NUM_WARPS (d_gtap_launch_config.warps_per_block)
#define GTAP_RUNTIME_TOTAL_WORKERS (d_gtap_launch_config.total_workers)
#define GTAP_RUNTIME_TASKS_PER_WORKER (d_gtap_launch_config.tasks_per_worker)
#define GTAP_RUNTIME_NUM_QUEUES (d_gtap_launch_config.num_queues)
#define GTAP_RUNTIME_QUEUE_CAPACITY (d_gtap_launch_config.queue_capacity)
#else
#define GTAP_RUNTIME_GRID_SIZE (gtap_stored_launch_config().grid_size)
#define GTAP_RUNTIME_BLOCK_SIZE (gtap_stored_launch_config().block_size)
#define GTAP_RUNTIME_NUM_WARPS (gtap_stored_launch_config().warps_per_block)
#define GTAP_RUNTIME_TOTAL_WORKERS (gtap_stored_launch_config().total_workers)
#define GTAP_RUNTIME_TASKS_PER_WORKER (gtap_stored_launch_config().tasks_per_worker)
#define GTAP_RUNTIME_NUM_QUEUES (gtap_stored_launch_config().num_queues)
#define GTAP_RUNTIME_QUEUE_CAPACITY (gtap_stored_launch_config().queue_capacity)
#endif
#define GTAP_RUNTIME_TOTAL_TASKS \
    (GTAP_RUNTIME_TOTAL_WORKERS * GTAP_RUNTIME_TASKS_PER_WORKER)
#define GTAP_RUNTIME_GLOBAL_QUEUE_CAPACITY \
    (GTAP_RUNTIME_TOTAL_WORKERS * GTAP_RUNTIME_QUEUE_CAPACITY)

inline size_t gtap_thread_gq_dynamic_shared_bytes(
    int block_size, int num_queues
) {
    const size_t warps = block_size / GTAP_WARP_SIZE;
    size_t bytes = sizeof(TaskContext) * warps;
    bytes = gtap_align_up(bytes, alignof(int));
    bytes += sizeof(int) * warps * num_queues;
    bytes += sizeof(int) * warps * num_queues * GTAP_WARP_SIZE;
    if (num_queues > 1) {
        bytes += sizeof(int) * warps * num_queues;
    }
#ifdef GTAP_PROFILE
    bytes += 2 * sizeof(int) * warps;
#endif
    return bytes;
}

__device__ __forceinline__ int* gtap_global_queue_slot(
    int queue_idx, int position
) {
    const size_t capacity =
        static_cast<size_t>(d_gtap_launch_config.total_workers) *
        d_gtap_launch_config.queue_capacity;
    return &d_global_task_queue[
        static_cast<size_t>(queue_idx) * capacity + position];
}

__device__ __forceinline__ int get_task_id_generated(
    int warp_id_global, int queue_idx, int idx
) {
    int offset =
        (warp_id_global * d_gtap_launch_config.num_queues + queue_idx) *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE +
        idx;
    return d_task_id_generated_by_queue_idx[offset];
}

__device__ __forceinline__ void set_task_id_generated(
    int warp_id_global, int queue_idx, int idx, int task_id
) {
    if (idx >= GTAP_TASK_ID_GEN_QUEUE_STRIDE) {
        GTAP_RECORD_GENERATED_TASK_ID_BUFFER_OVERFLOW(
            task_id, queue_idx, idx, GTAP_TASK_ID_GEN_QUEUE_STRIDE);
    }
    int offset =
        (warp_id_global * d_gtap_launch_config.num_queues + queue_idx) *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE +
        idx;
    d_task_id_generated_by_queue_idx[offset] = task_id;
}

static size_t __gtap_runtime_device_allocation_bytes() {
    const gtap_launch_config& c = gtap_stored_launch_config();
    const size_t workers = c.total_workers;
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * tasks;
    const size_t queue_metadata_bytes = 3 * sizeof(int) * c.num_queues;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = gtap_host_task_data_stride() * tasks;
    const size_t task_id_list_bytes = sizeof(TaskIdList) * workers;
    const size_t task_id_pool_bytes = 2 * sizeof(int) * tasks;
    const size_t task_id_generated_bytes = sizeof(int) * workers *
        c.num_queues * GTAP_TASK_ID_GEN_QUEUE_STRIDE;
    size_t total = global_queue_bytes + header_bytes + task_data_bytes +
        task_id_list_bytes + task_id_pool_bytes + task_id_generated_bytes +
        queue_metadata_bytes;
#ifdef GTAP_PROFILE
    total += workers * c.profile_capacity *
        (2 * sizeof(long long) + sizeof(int));
#endif
    return total;
}

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks =
        total_workers * runtime_config.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * total_tasks;

    #ifdef INIT_PROFILE
    printf("\n=== init_task_runtime (GQ) detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    
    cudaEventRecord(start);
    #endif

    int* d_global_task_queue_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_global_task_queue_ptr),
        global_queue_bytes));
    
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(global queue, %zu bytes): %.3f ms\n", global_queue_bytes, elapsed);
    cudaEventRecord(start);
    #endif
    
    GTAP_CUDA_TRY(cudaMemset(
        d_global_task_queue_ptr, 0, global_queue_bytes));
    
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(global queue, %zu bytes): %.3f ms\n", global_queue_bytes, elapsed);
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
    size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE *
        GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES *
        GTAP_TASK_ID_GEN_QUEUE_STRIDE;
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


    int* d_queue_head_ptr = nullptr;
    int* d_queue_tail_ptr = nullptr;
    int* d_queue_alloc_ptr = nullptr;
    const size_t queue_metadata_bytes =
        sizeof(int) * runtime_config.num_queues;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_head_ptr), queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_tail_ptr), queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_alloc_ptr), queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMemset(d_queue_head_ptr, 0, queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMemset(d_queue_tail_ptr, 0, queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMemset(d_queue_alloc_ptr, 0, queue_metadata_bytes));

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
        d_global_task_queue, &d_global_task_queue_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_head, &d_queue_head_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_tail, &d_queue_tail_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_alloc, &d_queue_alloc_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_global_task_queue): %.3f ms\n", elapsed);
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
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx): %.3f ms\n", elapsed);
    #endif

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    // Initialize d_active_worker_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(global state vars): %.3f ms\n", elapsed);
    #endif

#ifdef GTAP_PROFILE
    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    const size_t profile_workers = total_workers;
    const size_t profile_long_bytes = sizeof(long long) * profile_workers *
        gtap_profile_capacity();
    const size_t profile_int_bytes = sizeof(int) * profile_workers *
        gtap_profile_capacity();
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
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
    GTAP_CUDA_TRY(cudaMemset(having_task_time_ptr, 0, profile_long_bytes));
    GTAP_CUDA_TRY(cudaMemset(working_time_ptr, 0, profile_long_bytes));
    GTAP_CUDA_TRY(cudaMemset(tasks_processed_count_ptr, 0, profile_int_bytes));
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
    printf("=== init_task_runtime (GQ) profiling complete ===\n\n");
    #endif
    
    return cudaGetLastError();
}

cudaError_t __gtap_finalize_task_runtime() {
    int* d_global_task_queue_ptr = nullptr;
    int* d_queue_head_ptr = nullptr;
    int* d_queue_tail_ptr = nullptr;
    int* d_queue_alloc_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_head_ptr, d_queue_head, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_tail_ptr, d_queue_tail, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_alloc_ptr, d_queue_alloc, sizeof(int*)));
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

    
    // Free global queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_global_task_queue_ptr));
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
    if (d_queue_head_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(d_queue_head_ptr));
    if (d_queue_tail_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(d_queue_tail_ptr));
    if (d_queue_alloc_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(d_queue_alloc_ptr));
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
    }
#ifdef GTAP_PROFILE
    if (having_task_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(having_task_time_ptr));
    if (working_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(working_time_ptr));
    if (tasks_processed_count_ptr != nullptr)
        GTAP_CUDA_TRY(cudaFree(tasks_processed_count_ptr));
#endif
    
    GTAP_CUDA_TRY(gtap_finalize_runtime_error_report());

    return cudaGetLastError();
}

cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated = nullptr
) {
    GTAP_CUDA_TRY(gtap_validate_config(config));
    gtap_launch_config runtime_config{
        config.grid_size,
        config.block_size,
        config.block_size / GTAP_WARP_SIZE,
        config.grid_size * (config.block_size / GTAP_WARP_SIZE),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        config.profile_capacity_per_warp,
        gtap_thread_gq_dynamic_shared_bytes(
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

cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    return gtap_initialize(gtap_thread_config{}, device_bytes_allocated);
}

cudaError_t gtap_finalize() {
    cudaError_t err = __gtap_finalize_task_runtime();
    if (err == cudaSuccess) gtap_initialized_flag() = false;
    return err;
}

// Reset task runtime state for re-execution
cudaError_t __gtap_reset_task_runtime() {
    gtap_reset_runtime_error_report_host();

    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));

    
    // Clear global task queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemset(
            d_global_task_queue_ptr, 0,
            sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS));
    }
    
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
    
    // Clear task ID generated array
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE *
            GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE;
        GTAP_CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    }

    // Reset global state
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    // Reset queue head, tail, and alloc
    int* d_queue_head_ptr = nullptr;
    int* d_queue_tail_ptr = nullptr;
    int* d_queue_alloc_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_head_ptr, d_queue_head, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_tail_ptr, d_queue_tail, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_alloc_ptr, d_queue_alloc, sizeof(int*)));
    const size_t queue_metadata_bytes =
        sizeof(int) * GTAP_RUNTIME_NUM_QUEUES;
    GTAP_CUDA_TRY(cudaMemset(d_queue_head_ptr, 0, queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMemset(d_queue_tail_ptr, 0, queue_metadata_bytes));
    GTAP_CUDA_TRY(cudaMemset(d_queue_alloc_ptr, 0, queue_metadata_bytes));

    // Reset profile data if enabled
    #ifdef GTAP_PROFILE
    long long* having_ptr = nullptr;
    long long* working_ptr = nullptr;
    int* counts_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&having_ptr, having_task_time, sizeof(having_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&working_ptr, working_time, sizeof(working_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&counts_ptr, tasks_processed_count, sizeof(counts_ptr)));
    const size_t profile_workers = gtap_stored_launch_config().total_workers;
    GTAP_CUDA_TRY(cudaMemset(having_ptr, 0, sizeof(long long) * profile_workers * gtap_profile_capacity()));
    GTAP_CUDA_TRY(cudaMemset(working_ptr, 0, sizeof(long long) * profile_workers * gtap_profile_capacity()));
    GTAP_CUDA_TRY(cudaMemset(counts_ptr, 0, sizeof(int) * profile_workers * gtap_profile_capacity()));
    #endif
    
    // Reinitialize warp ID pools metadata
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0,
        gtap_stored_stream()>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    
    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}


#ifdef GTAP_PROFILE
cudaError_t get_warp_having_task_time_data(long long* host_having_task_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    return cudaMemcpy(host_having_task_time, ptr, sizeof(long long) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(host_working_time, ptr, sizeof(long long) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    int* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    return cudaMemcpy(host_counts, ptr, sizeof(int) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_having_task_time_data(int warp_global_id, long long* host_having_task_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(host_having_task_time,
        ptr + static_cast<size_t>(warp_global_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_working_time_data(int warp_global_id, long long* host_working_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(host_working_time,
        ptr + static_cast<size_t>(warp_global_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_single_warp_tasks_processed_count_data(int warp_global_id, int* host_counts, int max_samples) {
    int* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, tasks_processed_count, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(host_counts,
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

// ============================================================================
// Global Queue Operations (no steal needed - all workers pop from global queue)
// ============================================================================

// Keep the common case in warp-local shared memory.  Overflow remains in the
// private global staging buffer and is not published to the global queue until
// push_global_queue(), after the spawning lanes have finished task-data setup.
__device__ __forceinline__ void reserve_unpublished_task_id(
    TaskContext* ctx, int queue_idx, int task_id
) {
    int idx = atomicAdd(
        &ctx->task_id_generated_count_by_queue_idx[queue_idx], 1);
    if (idx < GTAP_WARP_SIZE) {
        ctx->staged_task_ids[queue_idx * GTAP_WARP_SIZE + idx] = task_id;
        return;
    }
    set_task_id_generated(
        get_warp_id_global(), queue_idx, idx - GTAP_WARP_SIZE, task_id);
}

__device__ __forceinline__ int get_unpublished_task_id(
    TaskContext* ctx, int queue_idx, int idx
) {
    if (idx < GTAP_WARP_SIZE)
        return ctx->staged_task_ids[queue_idx * GTAP_WARP_SIZE + idx];
    return get_task_id_generated(
        get_warp_id_global(), queue_idx, idx - GTAP_WARP_SIZE);
}

// Pop from global queue - returns number of tasks popped (up to max_count)
// Each lane gets a different task if available
template<TerminationMode M>
__device__ __forceinline__ int pop_global_queue(int* execute_task_id, int max_count, int queue_idx, bool prev_get_task) {
    int lane = get_lane_id();
    int count = 0;
    int base_head = 0;
    
    if (lane == 0) {
        // Try to claim slots from global queue
        while (true) {
            int old_head = load_L2(&d_queue_head[queue_idx]);
            int tail = load_L2(&d_queue_tail[queue_idx]);
            int available = max(0, tail - old_head);
            count = min(max_count, available);
            
            if (count == 0) break;
            
            // CAS to claim slots
            int new_head = old_head + count;
            if (atomicCAS(&d_queue_head[queue_idx], old_head, new_head) == old_head) {
                base_head = old_head;
                // Increment active worker count if this worker was previously idle
                if (M == TERMINATE_ON_ALL_TASKS_FINISH && !prev_get_task) {
                    atomicAdd(&d_active_worker_count, 1);
                }
                break;
            }
            // CAS failed, retry
        }
    }
    
    // Broadcast results to all lanes
    count = __shfl_sync(0xFFFFFFFFu, count, 0);
    base_head = __shfl_sync(0xFFFFFFFFu, base_head, 0);
    
    // Each lane reads its task (if it has one)
    if (lane < count) {
        int idx = (base_head + lane) %
            (d_gtap_launch_config.total_workers * d_gtap_launch_config.queue_capacity);
        int tid = load_L2(gtap_global_queue_slot(queue_idx, idx));
        *execute_task_id = tid;
#ifdef DEBUG
        printf("pop_global: tid=%d (queue %d) in lane %d\n", tid, queue_idx, lane);
#endif
    }
    
    return count;
}

// Push to global queue
template<TerminationMode M>
__device__ __forceinline__ void push_global_queue(
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
) {
    int lane = get_lane_id();
    // Calculate total generated tasks
    int all_generated_count = 0;
    int k_max = 0;
    int max_gen = -1;
    
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
    
    // Determine tasks to execute immediately vs push to queue
    *execute_task_count = max(0, min(GTAP_WARP_SIZE, max_gen));
    if (lane < *execute_task_count) {
        *execute_task_id = get_unpublished_task_id(ctx, k_max, lane);
#ifdef DEBUG
        printf("execute_immediately: tid=%d (queue %d) in lane %d\n", *execute_task_id, k_max, lane);
#endif
    }
    
    // Push remaining tasks to global queue
    #pragma unroll
    for (int kind = 0; kind < d_gtap_launch_config.num_queues; ++kind) {
        int first_idx_to_push = (kind == k_max) ? *execute_task_count : 0;
        int push_cnt = ctx->task_id_generated_count_by_queue_idx[kind] - first_idx_to_push;
        if (push_cnt <= 0) continue;
        
        // Reserve slots in global queue (allocate exclusive range)
        int base_pos = 0;
        if (lane == 0) {
            base_pos = atomicAdd(&d_queue_alloc[kind], push_cnt);
            // Overflow check
            int head_val = load_L2(&d_queue_head[kind]);
            if (base_pos + push_cnt - head_val > (d_gtap_launch_config.total_workers * d_gtap_launch_config.queue_capacity) - GTAP_QUEUE_MARGIN) {
            GTAP_RECORD_QUEUE_OVERFLOW(
                -1, kind, base_pos + push_cnt - head_val,
                (d_gtap_launch_config.total_workers * d_gtap_launch_config.queue_capacity) - GTAP_QUEUE_MARGIN);
            }
        }
        base_pos = __shfl_sync(0xFFFFFFFFu, base_pos, 0);
        
        // Write tasks to reserved slots
        for (int j = lane; j < push_cnt; j += GTAP_WARP_SIZE) {
            int tid = get_unpublished_task_id(
                ctx, kind, first_idx_to_push + j);
            int pos = (base_pos + j) % (d_gtap_launch_config.total_workers * d_gtap_launch_config.queue_capacity);
            store_L2(gtap_global_queue_slot(kind, pos), tid);
#ifdef DEBUG
            printf("push_global: tid=%d to queue %d, pos %d in lane %d\n", tid, kind, pos, lane);
#endif
        }
        __threadfence();
        __syncwarp();
        
        // Wait for prior commits and update tail (ensures in-order visibility)
        if (lane == 0) {
            while (load_L2(&d_queue_tail[kind]) != base_pos) {
                // spin - wait for prior pushers to commit
            }
            atomicAdd(&d_queue_tail[kind], push_cnt);
        }
    }
    __syncwarp();
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
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef DEBUG
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
    
    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);
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

    // Copy task data
    void* dest_task = __gtap_get_task_data(new_tid);
    memcpy(dest_task, task_data_ptr, task_data_size);
    
    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
}

// Push initial task to global queue
extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
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

    // Push to global queue (only warp 0, lane 0)
    if (warp_id_global == 0 && lane == 0) {
        store_L2(gtap_global_queue_slot(initial_queue_idx, 0), new_tid);
        __threadfence();
        store_L2(&d_queue_head[initial_queue_idx], 0);
        store_L2(&d_queue_alloc[initial_queue_idx], 1);
        store_L2(&d_queue_tail[initial_queue_idx], 1);
        __threadfence();
    }
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
    int* staged_task_ids = reinterpret_cast<int*>(shared_cursor);
    shared_cursor += sizeof(int) *
        d_gtap_launch_config.warps_per_block *
        d_gtap_launch_config.num_queues * GTAP_WARP_SIZE;
    int* queue_counts = d_gtap_launch_config.num_queues > 1
        ? reinterpret_cast<int*>(shared_cursor)
        : nullptr;

#ifdef GTAP_PROFILE
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
        warp_contexts[warp_id_in_block].staged_task_ids =
            staged_task_ids + warp_id_in_block *
                d_gtap_launch_config.num_queues * GTAP_WARP_SIZE;
        warp_contexts[warp_id_in_block].queue_idx = 0;
        warp_contexts[warp_id_in_block].id_list_free_pos_stale = d_gtap_launch_config.tasks_per_worker;
        #pragma unroll
        for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
            warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
        }
        if (warp_id_global == 0) {
#ifdef GTAP_PROFILE
            having_task_time[warp_id_global * gtap_profile_capacity()] = get_global_time();
#endif
            warp_contexts[0].id_list_alloc_pos = 1;
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
                    int head = load_L2(&d_queue_head[k]);
                    int tail = load_L2(&d_queue_tail[k]);
                    warp_queue_counts[k] = max(0, tail - head);
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
                int remaining = GTAP_WARP_SIZE - execute_task_count;
                int pop_count = pop_global_queue<M>(&execute_task_id, remaining, daq_idx, prev_get_task);
                execute_task_count += pop_count;
                if (execute_task_count != 0) break;
            }
            } else {
            int remaining = GTAP_WARP_SIZE - execute_task_count;
            int pop_count = pop_global_queue<M>(&execute_task_id, remaining, 0, prev_get_task);
            execute_task_count += pop_count;
            }
        }

        if (execute_task_count == 0) {
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            // Check if all queues are empty
                            bool all_tasks_finished = 1;
                            #pragma unroll
                            for (int k = 0; k < d_gtap_launch_config.num_queues; ++k) {
                                int head = load_L2(&d_queue_head[k]);
                                int tail = load_L2(&d_queue_tail[k]);
                                if (head < tail) {
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
                    having_task_time[warp_id_global * gtap_profile_capacity() + having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = false;
            
            // Check termination condition
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
                    having_task_time[warp_id_global * gtap_profile_capacity() + having_time_idx[warp_id_in_block]] = get_global_time();
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
                    working_time[warp_id_global * gtap_profile_capacity() + working_time_idx[warp_id_in_block]] = get_global_time();
                    tasks_processed_count[warp_id_global * gtap_profile_capacity() + working_time_idx[warp_id_in_block]] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                }
            }
#endif
            // Execute task
            void* task_data = __gtap_get_task_data(execute_task_id);
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &warp_contexts[warp_id_in_block]);
#ifdef DEBUG
            printf("executed_task_id: %d in lane %d of warp %d of block %d\n", execute_task_id, lane, warp_id_in_block, blockIdx.x);
#endif
            __threadfence();
        }
        __syncwarp();
#ifdef GTAP_PROFILE
        if (lane == 0) {
            if (working_time_idx[warp_id_in_block] < gtap_profile_capacity()) {
                working_time[warp_id_global * gtap_profile_capacity() + working_time_idx[warp_id_in_block]] = get_global_time();
                tasks_processed_count[warp_id_global * gtap_profile_capacity() + working_time_idx[warp_id_in_block]] = execute_task_count;
                working_time_idx[warp_id_in_block]++;
            }
        }
        __syncwarp();
#endif

        push_global_queue<M>(
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
