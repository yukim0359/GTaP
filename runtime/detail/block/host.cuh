#pragma once

#include "runtime.cuh"

namespace gtap::detail::block {
using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = stored_launch_config();
    const size_t workers = c.total_workers;
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t queue_metadata_bytes = sizeof(BlockTaskQueueMetadata) * workers;
    const size_t queue_storage_bytes = sizeof(int) * tasks;
    const size_t task_id_free_position_bytes = sizeof(int) * workers;
    const size_t task_id_storage_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = host_task_data_stride() * tasks;
    const size_t entry_result_bytes =
        __gtap_auto_entry_result_size * static_cast<size_t>(c.block_size);
    size_t total =
        queue_metadata_bytes + queue_storage_bytes + task_id_free_position_bytes +
        task_id_storage_bytes + header_bytes + task_data_bytes +
        entry_result_bytes;
#ifdef GTAP_ENABLE_PROFILING
    total += sizeof(long long) * workers * profile_capacity();
    total += sizeof(unsigned long long) * workers;
#endif
    return total;
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    BlockTaskQueueMetadata* d_block_task_queue_metadata_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_block_task_queue_metadata_ptr), sizeof(BlockTaskQueueMetadata) * total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_block_task_queue_metadata_ptr, 0, sizeof(BlockTaskQueueMetadata) * total_workers, streams[0]));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_block_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_block_task_queue_storage_ptr, 0, sizeof(int) * total_tasks,
        streams[0]));

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_list_free_positions_ptr),
        sizeof(int) * total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_task_id_list_free_positions_ptr, 0, sizeof(int) * total_workers,
        streams[1]));
    int* d_task_id_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks, streams[2]));

    // Allocate static storage for task data (type-erased as byte array)
    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = host_task_data_stride();
    size_t task_data_size = max_task_size * total_tasks;
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

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_block_task_queue_metadata, &d_block_task_queue_metadata_ptr, sizeof(BlockTaskQueueMetadata*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_block_task_queue_storage, &d_block_task_queue_storage_ptr,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_list_free_positions, &d_task_id_list_free_positions_ptr,
        sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_entry_result_bytes, &d_entry_result_bytes_ptr,
        sizeof(char*)));
    GTAP_DETAIL_CUDA_TRY(init_device_task_data_stride());

#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    const size_t profile_bytes =
        sizeof(long long) * total_workers * profile_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(
        &profile_dropped_events_ptr),
        sizeof(unsigned long long) * total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_bytes, streams[1]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * total_workers, streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[1]));
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    return cudaDeviceSynchronize();
}

cudaError_t finalize_runtime() {
    // Get device pointers from symbols
    BlockTaskQueueMetadata* d_block_task_queue_metadata_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queue_metadata_ptr, d_block_task_queue_metadata, sizeof(BlockTaskQueueMetadata*)));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_block_task_queue_storage_ptr, d_block_task_queue_storage,
        sizeof(int*)));

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    int* d_task_id_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    char* d_entry_result_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_entry_result_bytes_ptr, d_entry_result_bytes,
        sizeof(char*)));
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
#endif

    // Free allocated memory
    if (d_block_task_queue_metadata_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_block_task_queue_metadata_ptr));
    }
    if (d_block_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_block_task_queue_storage_ptr));
    }

    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_list_free_positions_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
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
// This function clears all runtime state without reallocating memory
// Call this before each execution after the initial init_task_runtime call
cudaError_t reset_runtime() {
    reset_runtime_error_record_host();
    const launch_config& runtime_config = stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    BlockTaskQueueMetadata* d_block_task_queue_metadata_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queue_metadata_ptr, d_block_task_queue_metadata, sizeof(BlockTaskQueueMetadata*)));
    int* d_block_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_block_task_queue_storage_ptr, d_block_task_queue_storage,
        sizeof(int*)));

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

    // Clear task queues
    if (d_block_task_queue_metadata_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_block_task_queue_metadata_ptr, 0, sizeof(BlockTaskQueueMetadata) * total_workers,
            streams[0]));
    }
    if (d_block_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_block_task_queue_storage_ptr, 0, sizeof(int) * total_tasks,
            streams[0]));
    }

    // Reset task ID pool metadata.
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_id_list_free_positions_ptr, 0, sizeof(int) * total_workers,
            streams[1]));
    }
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks, streams[2]));
    }

    // Clear task data
    size_t max_task_size = host_task_data_stride();
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * total_tasks;
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    }
    if (d_entry_result_bytes_ptr != nullptr) {
        const size_t entry_result_size =
            __gtap_auto_entry_result_size *
            static_cast<size_t>(runtime_config.block_size);
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_entry_result_bytes_ptr, 0, entry_result_size, streams[3]));
    }

    // Reset profile data if enabled
#ifdef GTAP_ENABLE_PROFILING
    long long* working_time_ptr = nullptr;
    unsigned long long* profile_dropped_events_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &profile_dropped_events_ptr, profile_dropped_events,
        sizeof(profile_dropped_events_ptr)));
    const size_t profile_bytes =
        sizeof(long long) * total_workers * profile_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_bytes, streams[1]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * total_workers, streams[0]));
#endif

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    // Reset global state
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)));

    // Reinitialize block ID pools metadata
    init_block_id_pools_metadata<<<runtime_config.grid_size, 1>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());

    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    return cudaGetLastError();
}


}  // namespace gtap::detail::block
