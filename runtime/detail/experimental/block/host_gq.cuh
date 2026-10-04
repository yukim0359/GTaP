#pragma once

#include "runtime_gq.cuh"

namespace gtap::detail::block {
using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = h_launch_config;
    const size_t tasks =
        static_cast<size_t>(c.total_workers) * c.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * tasks;
    const size_t task_id_free_position_bytes =
        sizeof(int) * c.total_workers;
    const size_t task_id_pool_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = host_task_data_stride() * tasks;
    const size_t entry_result_bytes =
        __gtap_auto_entry_result_size * static_cast<size_t>(c.block_size);
    const size_t task_id_generated_bytes =
        sizeof(int) * c.total_workers * GTAP_MAX_CHILD_TASKS;
    size_t total = global_queue_bytes + task_id_free_position_bytes +
        header_bytes + task_data_bytes + entry_result_bytes +
        task_id_generated_bytes + task_id_pool_bytes;
#ifdef GTAP_ENABLE_PROFILING
    total += sizeof(long long) * c.total_workers * profile_timestamp_capacity();
    total += sizeof(unsigned long long) * c.total_workers;
#endif
    return total;
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = h_launch_config;
    const size_t total_tasks =
        static_cast<size_t>(runtime_config.total_workers) *
        runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    int* d_global_task_queue_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_global_task_queue_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_global_task_queue_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    
    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_list_free_positions_ptr),
        sizeof(int) * GTAP_RUNTIME_GRID_SIZE));
    // Lazy initialization: set free positions to -1; they are initialized later.
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_task_id_list_free_positions_ptr, 0xFF,
        sizeof(int) * GTAP_RUNTIME_GRID_SIZE, streams[1]));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, streams[2]));

    // Allocate static storage for task data (type-erased as byte array)
    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = host_task_data_stride();
    size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));

    char* d_entry_result_bytes_ptr = nullptr;
    const size_t entry_result_size =
        __gtap_auto_entry_result_size *
        static_cast<size_t>(runtime_config.block_size);
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_entry_result_bytes_ptr),
        entry_result_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_entry_result_bytes_ptr, 0, entry_result_size, streams[3]));
    
    int* d_task_id_generated_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_ptr), task_id_array_size));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));

    int* d_task_id_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));

    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_global_task_queue, &d_global_task_queue_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &d_task_id_list_free_positions_ptr,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_entry_result_bytes, &d_entry_result_bytes_ptr,
        sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(init_device_task_data_stride());
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated, &d_task_id_generated_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    
#ifdef GTAP_ENABLE_PROFILING
    const size_t profile_bytes = sizeof(long long) *
        runtime_config.total_workers * profile_timestamp_capacity();
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&profile_dropped_events_ptr),
        sizeof(unsigned long long) * runtime_config.total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(working_time_ptr, 0, profile_bytes, streams[1]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * runtime_config.total_workers, streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[1]));
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    unsigned int uzero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_head, &uzero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_tail, &uzero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_alloc, &uzero, sizeof(unsigned int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)));
    
    init_block_id_pools_metadata<<<
        runtime_config.grid_size, 1, 0, h_stream>>>();
    return cudaDeviceSynchronize();
}

cudaError_t finalize_runtime() {
    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    
    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    char* d_entry_result_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_entry_result_bytes_ptr, d_entry_result_bytes,
        sizeof(char*)));
    
    int* d_task_id_generated_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
#endif

    
    // Free global queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_global_task_queue_ptr));
    }
    
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_list_free_positions_ptr));
    }
    
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    if (d_entry_result_bytes_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_entry_result_bytes_ptr));
    }
    
    if (d_task_id_generated_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_generated_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
#ifdef GTAP_ENABLE_PROFILING
    if (working_time_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
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

    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    
    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    char* d_entry_result_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_entry_result_bytes_ptr, d_entry_result_bytes,
        sizeof(char*)));
    
    int* d_task_id_generated_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));

    
    // Clear global task queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_global_task_queue_ptr, 0,
            sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS, streams[0]));
    }
    
    // Reset task ID free positions (0xFF = -1 for lazy initialization).
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_id_list_free_positions_ptr, 0xFF,
            sizeof(int) * GTAP_RUNTIME_GRID_SIZE, streams[1]));
    }
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, streams[2]));
    }
    
    // Clear task data
    size_t max_task_size = host_task_data_stride();
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    }
    if (d_entry_result_bytes_ptr != nullptr) {
        const size_t entry_result_size =
            __gtap_auto_entry_result_size *
            static_cast<size_t>(h_launch_config.block_size);
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_entry_result_bytes_ptr, 0, entry_result_size, streams[3]));
    }
    
    // Clear task ID generated array
    if (d_task_id_generated_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
    }

    // Reset profile data if enabled
#ifdef GTAP_ENABLE_PROFILING
    long long* working_ptr = nullptr;
    unsigned long long* dropped_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&working_ptr, working_time, sizeof(working_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &dropped_ptr, profile_dropped_events, sizeof(dropped_ptr)));
    const size_t profile_bytes = sizeof(long long) *
        h_launch_config.total_workers * profile_timestamp_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(working_ptr, 0, profile_bytes, streams[1]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        dropped_ptr, 0,
        sizeof(unsigned long long) * h_launch_config.total_workers,
        streams[0]));
#endif
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }
    
    // Reset global state
    int zero = 0;
    unsigned int uzero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_head, &uzero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_tail, &uzero, sizeof(unsigned int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_queue_alloc, &uzero, sizeof(unsigned int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)));
    
    // Reinitialize block ID pools metadata
    init_block_id_pools_metadata<<<GTAP_RUNTIME_GRID_SIZE, 1>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    
    return cudaGetLastError();
}


}  // namespace gtap::detail::block
