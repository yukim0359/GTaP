#pragma once

#include "gtap_runtime_thread_gq.cuh"

struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    int max_tasks_per_warp = 10000;
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
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
#ifdef GTAP_ENABLE_PROFILING
    if (config.profile_capacity_per_warp <= 0 ||
        config.profile_capacity_per_warp > INT_MAX / 2) {
        return cudaErrorInvalidValue;
    }
#endif
    return cudaSuccess;
}

namespace gtap::detail::thread {
using namespace gtap::detail;

inline size_t dynamic_shared_bytes(
    int block_size, int num_queues
) {
    const size_t warps = block_size / GTAP_WARP_SIZE;
    size_t bytes = sizeof(TaskContext) * warps;
    bytes = align_up(bytes, alignof(int));
    bytes += sizeof(int) * warps * num_queues;
    bytes += sizeof(int) * warps * num_queues * GTAP_WARP_SIZE;
    if (num_queues > 1) {
        bytes += sizeof(int) * warps * num_queues;
    }
#ifdef GTAP_ENABLE_PROFILING
    bytes += sizeof(int) * warps;
#endif
    return bytes;
}

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = stored_launch_config();
    const size_t workers = c.total_workers;
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * tasks;
    const size_t queue_metadata_bytes = 3 * sizeof(int) * c.num_queues;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = host_task_data_stride() * tasks;
    const size_t task_id_free_position_bytes = sizeof(int) * workers;
    const size_t task_id_pool_bytes = 2 * sizeof(int) * tasks;
    const size_t task_id_generated_bytes = sizeof(int) * workers *
        c.num_queues * GTAP_TASK_ID_GEN_QUEUE_STRIDE;
    size_t total = global_queue_bytes + header_bytes + task_data_bytes +
        task_id_free_position_bytes + task_id_pool_bytes + task_id_generated_bytes +
        queue_metadata_bytes;
#ifdef GTAP_ENABLE_PROFILING
    total += workers * profile_capacity() *
             (sizeof(long long) + sizeof(int));
    total += workers * sizeof(unsigned long long);
#endif
    return total;
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks =
        total_workers * runtime_config.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * total_tasks;

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== init_task_runtime (GQ) detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    
    cudaEventRecord(start);
    #endif

    int* d_global_task_queue_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_global_task_queue_ptr),
        global_queue_bytes));
    
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(global queue, %zu bytes): %.3f ms\n", global_queue_bytes, elapsed);
    cudaEventRecord(start);
    #endif
    
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        d_global_task_queue_ptr, 0, global_queue_bytes));
    
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(global queue, %zu bytes): %.3f ms\n", global_queue_bytes, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, elapsed);
    cudaEventRecord(start);
    #endif

    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = host_task_data_stride();
    size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);  
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_list_free_positions_ptr),
        sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task ID free positions, %zu bytes): %.3f ms\n",
           sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS,
           elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        d_task_id_list_free_positions_ptr, 0xFF,
        sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(Task ID free positions, %zu bytes): %.3f ms\n",
           sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS,
           elapsed);
    cudaEventRecord(start);
    #endif

    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE *
        GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES *
        GTAP_TASK_ID_GEN_QUEUE_STRIDE;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_by_queue_idx_ptr), task_id_array_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(task_id_generated, %zu bytes): %.3f ms\n", task_id_array_size, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
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
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_head_ptr), queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_tail_ptr), queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_queue_alloc_ptr), queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_head_ptr, 0, queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_tail_ptr, 0, queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_alloc_ptr, 0, queue_metadata_bytes));

    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_valid_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        d_task_id_valid_ptr, 0, sizeof(int) * total_tasks));

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_global_task_queue, &d_global_task_queue_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_head, &d_queue_head_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_tail, &d_queue_tail_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_queue_alloc, &d_queue_alloc_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_global_task_queue): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_headers): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(init_device_task_data_stride());
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_data_bytes): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &d_task_id_list_free_positions_ptr,
        sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_list_free_positions): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx, &d_task_id_generated_by_queue_idx_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx): %.3f ms\n", elapsed);
    #endif

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    // Initialize d_active_warp_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));
    
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(global state vars): %.3f ms\n", elapsed);
    #endif

#ifdef GTAP_ENABLE_PROFILING
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    const size_t profile_workers = total_workers;
    const size_t profile_long_bytes = sizeof(long long) * profile_workers *
        profile_capacity();
    const size_t profile_int_bytes = sizeof(int) * profile_workers *
        profile_capacity();
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_long_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr), profile_int_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&profile_dropped_events_ptr),
        sizeof(unsigned long long) * profile_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(working_time_ptr, 0, profile_long_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(tasks_processed_count_ptr, 0, profile_int_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * profile_workers));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
#endif

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0,
        stored_stream()>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    #ifdef GTAP_INTERNAL_PROFILE_INIT
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

cudaError_t finalize_runtime() {
    int* d_global_task_queue_ptr = nullptr;
    int* d_queue_head_ptr = nullptr;
    int* d_queue_tail_ptr = nullptr;
    int* d_queue_alloc_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_head_ptr, d_queue_head, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_tail_ptr, d_queue_tail, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_alloc_ptr, d_queue_alloc, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
#endif

    
    // Free global queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_global_task_queue_ptr));
    }
    
    // Free other allocated memory
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_list_free_positions_ptr));
    }
    
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_generated_by_queue_idx_ptr));
    }
    if (d_queue_head_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(d_queue_head_ptr));
    if (d_queue_tail_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(d_queue_tail_ptr));
    if (d_queue_alloc_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(d_queue_alloc_ptr));
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
    }
#ifdef GTAP_ENABLE_PROFILING
    if (working_time_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
    if (tasks_processed_count_ptr != nullptr)
        GTAP_DETAIL_CUDA_TRY(cudaFree(tasks_processed_count_ptr));
    if (profile_dropped_events_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(profile_dropped_events_ptr));
    }
#endif
    
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());

    return cudaGetLastError();
}

// Reset task runtime state for re-execution
cudaError_t reset_runtime() {
    reset_runtime_error_record_host();

    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));

    
    // Clear global task queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemset(
            d_global_task_queue_ptr, 0,
            sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS));
    }
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    }
    
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t max_task_size = host_task_data_stride();
        size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
        GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    }
    
    // Reset task ID free positions.
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemset(
            d_task_id_list_free_positions_ptr, 0xFF,
            sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
    }
    
    // Clear task ID generated array
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE *
            GTAP_RUNTIME_NUM_WARPS * GTAP_RUNTIME_NUM_QUEUES *
            GTAP_TASK_ID_GEN_QUEUE_STRIDE;
        GTAP_DETAIL_CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    }

    // Reset global state
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));
    
    // Reset queue head, tail, and alloc
    int* d_queue_head_ptr = nullptr;
    int* d_queue_tail_ptr = nullptr;
    int* d_queue_alloc_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_head_ptr, d_queue_head, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_tail_ptr, d_queue_tail, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_queue_alloc_ptr, d_queue_alloc, sizeof(int*)));
    const size_t queue_metadata_bytes =
        sizeof(int) * GTAP_RUNTIME_NUM_QUEUES;
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_head_ptr, 0, queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_tail_ptr, 0, queue_metadata_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(d_queue_alloc_ptr, 0, queue_metadata_bytes));

    // Reset profile data if enabled
    #ifdef GTAP_ENABLE_PROFILING
    long long* working_ptr = nullptr;
    int* counts_ptr = nullptr;
    unsigned long long* dropped_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&working_ptr, working_time, sizeof(working_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&counts_ptr, tasks_processed_count, sizeof(counts_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &dropped_ptr, profile_dropped_events, sizeof(dropped_ptr)));
    const size_t profile_workers = stored_launch_config().total_workers;
    GTAP_DETAIL_CUDA_TRY(cudaMemset(working_ptr, 0, sizeof(long long) * profile_workers * profile_capacity()));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(counts_ptr, 0, sizeof(int) * profile_workers * profile_capacity()));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        dropped_ptr, 0, sizeof(unsigned long long) * profile_workers));
    #endif
    
    // Reinitialize warp ID pools metadata
    const launch_config& runtime_config = stored_launch_config();
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0,
        stored_stream()>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    
    return cudaGetLastError();
}


}  // namespace gtap::detail::thread

cudaError_t gtap_initialize(
    const gtap_thread_config& config,
    size_t* device_bytes_allocated = nullptr
) {
    GTAP_DETAIL_CUDA_TRY(gtap_validate_config(config));
    gtap::detail::launch_config runtime_config{
        config.grid_size,
        config.block_size,
        config.block_size / GTAP_WARP_SIZE,
        config.grid_size * (config.block_size / GTAP_WARP_SIZE),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        config.profile_capacity_per_warp,
        gtap::detail::thread::dynamic_shared_bytes(
            config.block_size, config.num_queues)
    };
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(runtime_config));
    gtap::detail::stored_stream() = config.stream;
    cudaError_t err = gtap::detail::thread::initialize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = true;
        if (device_bytes_allocated != nullptr) {
            *device_bytes_allocated = gtap::detail::thread::runtime_device_allocation_bytes();
        }
    }
    return err;
}

cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    return gtap_initialize(gtap_thread_config{}, device_bytes_allocated);
}

cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::thread::finalize_runtime();
    if (err == cudaSuccess) gtap::detail::initialized_flag() = false;
    return err;
}

cudaError_t gtap_reset() {
    return gtap::detail::thread::reset_runtime();
}
