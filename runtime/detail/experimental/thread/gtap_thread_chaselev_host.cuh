#pragma once

#include "gtap_runtime_thread_chaselev.cuh"

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
    return shared_layout_for(
        block_size / GTAP_WARP_SIZE, num_queues, true).bytes;
}

static size_t runtime_device_allocation_bytes() {
    const size_t queue_ptr_array_bytes = sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES;
    const size_t queue_plane_bytes =
        (size_t)GTAP_RUNTIME_NUM_QUEUES * sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS;
    const size_t header_bytes = sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_data_bytes = host_task_data_stride() * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_id_free_position_bytes =
        sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS;
    const size_t queue_storage_bytes =
        sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS;
    const size_t task_id_pool_bytes =
        2 * sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS;
    size_t total =
        queue_ptr_array_bytes + queue_plane_bytes + header_bytes + task_data_bytes +
           task_id_free_position_bytes +
           queue_storage_bytes + task_id_pool_bytes;
#ifdef GTAP_ENABLE_PROFILING
    total += GTAP_RUNTIME_TOTAL_WORKERS * profile_capacity() *
        (sizeof(long long) + sizeof(int));
    total += GTAP_RUNTIME_TOTAL_WORKERS * sizeof(unsigned long long);
#endif
    return total;
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = stored_launch_config();
    const size_t total_tasks =
        static_cast<size_t>(runtime_config.total_workers) *
        runtime_config.tasks_per_worker;

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    
    cudaEventRecord(start);
    #endif

    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queue_metadata_ptrptr), sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES));
    
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES, elapsed);
    #endif
    
    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES));
    for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(start);
        #endif
        WarpTaskQueueMetadata* plane_ptr = nullptr;
        GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&plane_ptr), sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_DETAIL_CUDA_TRY(cudaMemset(plane_ptr, 0, sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemset(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS, elapsed);
        #endif
        h_warp_task_queue_metadata_planes[k] = plane_ptr;
    }
    
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(d_warp_task_queue_metadata_ptrptr, h_warp_task_queue_metadata_planes, sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyHostToDevice));

    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_warp_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES, elapsed);
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

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queue_metadata, &d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata**)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &d_warp_task_queue_storage_ptr,
        sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_warp_task_queue_metadata): %.3f ms\n", elapsed);
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
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  initialized task-id storage: %.3f ms\n", elapsed);
    #endif

    free(h_warp_task_queue_metadata_planes);

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
    #endif
    // Initialize d_active_warp_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_warp_count): %.3f ms\n", elapsed);
    #endif

#ifdef GTAP_ENABLE_PROFILING
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    const size_t profile_time_bytes =
        sizeof(long long) * runtime_config.total_workers * profile_capacity();
    const size_t profile_count_bytes =
        sizeof(int) * runtime_config.total_workers * profile_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr),
        profile_time_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr),
        profile_count_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&profile_dropped_events_ptr),
        sizeof(unsigned long long) * runtime_config.total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr,
        sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        working_time_ptr, 0, profile_time_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        tasks_processed_count_ptr, 0, profile_count_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * runtime_config.total_workers));
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
    printf("=== init_task_runtime profiling complete ===\n\n");
    #endif
    
    return cudaGetLastError();
}

cudaError_t finalize_runtime() {
    // Get device pointers from symbols
    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage,
        sizeof(int*)));
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
    
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time,
        sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
#endif

    
    // Get queue plane pointers from device
    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES));
    if (d_warp_task_queue_metadata_ptrptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyDeviceToHost));
        
        // Free each queue plane
        for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
            if (h_warp_task_queue_metadata_planes[k] != nullptr) {
                GTAP_DETAIL_CUDA_TRY(cudaFree(h_warp_task_queue_metadata_planes[k]));
            }
        }
    }
    free(h_warp_task_queue_metadata_planes);
    
    // Free queue pointer array
    if (d_warp_task_queue_metadata_ptrptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_warp_task_queue_metadata_ptrptr));
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_warp_task_queue_storage_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
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
    
#ifdef GTAP_ENABLE_PROFILING
    if (working_time_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
    }
    if (tasks_processed_count_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(tasks_processed_count_ptr));
    }
    if (profile_dropped_events_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(profile_dropped_events_ptr));
    }
#endif

    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());

    return cudaGetLastError();
}

// Reset task runtime state for re-execution
// This function clears all runtime state without reallocating memory
// Call this before each execution after the initial init_task_runtime call
cudaError_t reset_runtime() {
    reset_runtime_error_record_host();
    const launch_config& runtime_config =
        stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks =
        total_workers * runtime_config.tasks_per_worker;

    // Get device pointers from symbols
    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage,
        sizeof(int*)));
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
    
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    int* tasks_processed_count_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time,
        sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &tasks_processed_count_ptr, tasks_processed_count,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
#endif

    
    // Get queue plane pointers from device
    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * GTAP_RUNTIME_NUM_QUEUES, cudaMemcpyDeviceToHost));
    
    // Clear task queues
    for (int k = 0; k < GTAP_RUNTIME_NUM_QUEUES; ++k) {
        if (h_warp_task_queue_metadata_planes[k] != nullptr) {
            GTAP_DETAIL_CUDA_TRY(cudaMemset(h_warp_task_queue_metadata_planes[k], 0, sizeof(WarpTaskQueueMetadata) * GTAP_RUNTIME_GRID_SIZE * GTAP_RUNTIME_NUM_WARPS));
        }
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemset(
            d_warp_task_queue_storage_ptr, 0,
            sizeof(int) * total_tasks));
    }
    free(h_warp_task_queue_metadata_planes);
    
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
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemset(
            d_task_id_valid_ptr, 0, sizeof(int) * total_tasks));
    }
    
    // Reset global state
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));

    // Reset profile data if enabled
    #ifdef GTAP_ENABLE_PROFILING
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        working_time_ptr, 0,
        sizeof(long long) * total_workers * profile_capacity()));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        tasks_processed_count_ptr, 0,
        sizeof(int) * total_workers * profile_capacity()));
    GTAP_DETAIL_CUDA_TRY(cudaMemset(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * total_workers));
    #endif
    
    // Reinitialize warp ID pools metadata
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
    if (gtap::detail::initialized_flag()) return cudaErrorInitializationError;
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

cudaError_t gtap_finalize() {
    cudaError_t err = gtap::detail::thread::finalize_runtime();
    if (err == cudaSuccess) {
        gtap::detail::initialized_flag() = false;
        gtap::detail::stored_stream() = nullptr;
    }
    return err;
}

cudaError_t gtap_reset() {
    return gtap::detail::thread::reset_runtime();
}
