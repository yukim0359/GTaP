#pragma once

#include "gtap_runtime_thread.cuh"

struct gtap_thread_config {
    int grid_size = 4096;
    int block_size = 32;
    int max_tasks_per_warp = 10000;
    int num_queues = 1;
    int profile_capacity_per_warp = 15000;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_thread_config& config) {
    if (config.grid_size <= 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.block_size <= 0 ||
        config.block_size > GTAP_MAX_THREADS_PER_BLOCK ||
        config.block_size % gtap::detail::warp_size != 0) {
        return cudaErrorInvalidConfiguration;
    }
    if (config.max_tasks_per_warp <= 0 ||
        config.num_queues <= 0 ||
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
        block_size / warp_size, num_queues, true).bytes;
}

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = stored_launch_config();
    const size_t workers = static_cast<size_t>(c.total_workers);
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t queue_ptr_array_bytes = sizeof(WarpTaskQueueMetadata*) * c.num_queues;
    const size_t queue_metadata_bytes =
        static_cast<size_t>(c.num_queues) * sizeof(WarpTaskQueueMetadata) * workers;
    const size_t queue_storage_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = host_task_data_stride() * tasks;
    const size_t task_id_free_position_bytes = sizeof(int) * workers;
    const size_t task_id_storage_bytes = 2 * sizeof(int) * tasks;
    size_t total = queue_ptr_array_bytes + queue_metadata_bytes + queue_storage_bytes +
           header_bytes + task_data_bytes + task_id_free_position_bytes +
           task_id_storage_bytes;
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
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    const int NUM_STREAMS = runtime_config.num_queues + 3;
    cudaStream_t* streams = reinterpret_cast<cudaStream_t*>(
        malloc(sizeof(cudaStream_t) * NUM_STREAMS));
    if (streams == nullptr) return cudaErrorMemoryAllocation;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    cudaEventRecord(start);
    #endif

    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queue_metadata_ptrptr), sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues));

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues, elapsed);
    #endif

    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues));
    for (int k = 0; k < runtime_config.num_queues; ++k) {
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(start);
        #endif
        WarpTaskQueueMetadata* plane_ptr = nullptr;
        GTAP_DETAIL_CUDA_TRY(cudaMalloc(
            reinterpret_cast<void**>(&plane_ptr),
            sizeof(WarpTaskQueueMetadata) * total_workers));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * total_workers, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            plane_ptr, 0, sizeof(WarpTaskQueueMetadata) * total_workers, streams[k]));
        #ifdef GTAP_INTERNAL_PROFILE_INIT
        cudaEventRecord(stop, streams[k]);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemsetAsync(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueueMetadata) * total_workers, elapsed);
        #endif
        h_warp_task_queue_metadata_planes[k] = plane_ptr;
    }

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(d_warp_task_queue_metadata_ptrptr, h_warp_task_queue_metadata_planes, sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues, cudaMemcpyHostToDevice));

    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_warp_task_queue_storage_ptr),
        sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * total_tasks));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n",
           sizeof(TaskHeader) * total_tasks, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
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
    size_t max_task_size = host_task_data_stride();
    size_t task_data_size = max_task_size * total_tasks;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[runtime_config.num_queues + 1]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, streams[runtime_config.num_queues + 1]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_list_free_positions_ptr),
        sizeof(int) * total_workers));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task ID free positions, %zu bytes): %.3f ms\n",
           sizeof(int) * total_workers, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_task_id_list_free_positions_ptr, 0xFF, sizeof(int) * total_workers,
        streams[runtime_config.num_queues + 2]));
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr), sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_valid_ptr), sizeof(int) * total_tasks));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        d_task_id_valid_ptr, 0, sizeof(int) * total_tasks,
        streams[runtime_config.num_queues + 2]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, streams[runtime_config.num_queues + 2]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(Task ID free positions, %zu bytes): %.3f ms\n",
           sizeof(int) * total_workers, elapsed);
    cudaEventRecord(start);
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queue_metadata, &d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata**)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &d_warp_task_queue_storage_ptr, sizeof(int*)));
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
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_valid, &d_task_id_valid_ptr, sizeof(int*)));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_list_free_positions): %.3f ms\n", elapsed);
    cudaEventRecord(start);
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
    const size_t profile_long_bytes =
        sizeof(long long) * total_workers * profile_capacity();
    const size_t profile_int_bytes =
        sizeof(int) * total_workers * profile_capacity();
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_long_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&tasks_processed_count_ptr), profile_int_bytes));
    GTAP_DETAIL_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(
        &profile_dropped_events_ptr),
        sizeof(unsigned long long) * total_workers));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        tasks_processed_count, &tasks_processed_count_ptr,
        sizeof(tasks_processed_count_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        profile_dropped_events, &profile_dropped_events_ptr,
        sizeof(profile_dropped_events_ptr)));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0, profile_long_bytes, streams[1 % NUM_STREAMS]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0, profile_int_bytes,
        streams[2 % NUM_STREAMS]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * total_workers, streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[1 % NUM_STREAMS]));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[2 % NUM_STREAMS]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size,
        runtime_config.warps_per_block * warp_size>>>();
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
    const int num_queues = stored_launch_config().num_queues;
    // Get device pointers from symbols
    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    int* d_task_id_storage_ptr = nullptr;
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
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

    // Get queue plane pointers from device
    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * num_queues));
    if (d_warp_task_queue_metadata_ptrptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * num_queues, cudaMemcpyDeviceToHost));

        // Free each queue plane
        for (int k = 0; k < num_queues; ++k) {
            if (h_warp_task_queue_metadata_planes[k] != nullptr) {
                GTAP_DETAIL_CUDA_TRY(cudaFree(h_warp_task_queue_metadata_planes[k]));
            }
        }
    }
    free(h_warp_task_queue_metadata_planes);

    if (d_warp_task_queue_metadata_ptrptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_warp_task_queue_metadata_ptrptr));
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_warp_task_queue_storage_ptr));
    }

    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }

    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }

    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_list_free_positions_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaFree(d_task_id_valid_ptr));
    }
#ifdef GTAP_ENABLE_PROFILING
    if (working_time_ptr != nullptr) GTAP_DETAIL_CUDA_TRY(cudaFree(working_time_ptr));
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
    const launch_config& runtime_config = stored_launch_config();
    const size_t total_workers = runtime_config.total_workers;
    const size_t total_tasks = total_workers * runtime_config.tasks_per_worker;

    const int NUM_STREAMS = runtime_config.num_queues + 3;
    cudaStream_t* streams = reinterpret_cast<cudaStream_t*>(
        malloc(sizeof(cudaStream_t) * NUM_STREAMS));
    if (streams == nullptr) return cudaErrorMemoryAllocation;
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    int* d_task_id_list_free_positions_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_list_free_positions_ptr, d_task_id_list_free_positions,
        sizeof(int*)));
    int* d_task_id_valid_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_valid_ptr, d_task_id_valid, sizeof(int*)));
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

    // Get queue plane pointers from device
    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues, cudaMemcpyDeviceToHost));

    // Clear task queues
    for (int k = 0; k < runtime_config.num_queues; ++k) {
        if (h_warp_task_queue_metadata_planes[k] != nullptr) {
            GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
                h_warp_task_queue_metadata_planes[k], 0,
                sizeof(WarpTaskQueueMetadata) * total_workers, streams[k]));
        }
    }
    if (d_warp_task_queue_storage_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_warp_task_queue_storage_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    }
    free(h_warp_task_queue_metadata_planes);

    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_headers_ptr, 0, sizeof(TaskHeader) * total_tasks,
            streams[runtime_config.num_queues]));
    }

    size_t max_task_size = host_task_data_stride();
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * total_tasks;
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[runtime_config.num_queues + 1]));
    }

    // Reset task ID free positions.
    if (d_task_id_list_free_positions_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_id_list_free_positions_ptr, 0xFF, sizeof(int) * total_workers,
            streams[runtime_config.num_queues + 2]));
    }
    if (d_task_id_valid_ptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
            d_task_id_valid_ptr, 0, sizeof(int) * total_tasks,
            streams[runtime_config.num_queues + 2]));
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
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        working_time_ptr, 0,
        sizeof(long long) * total_workers * profile_capacity(),
        streams[1 % NUM_STREAMS]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        tasks_processed_count_ptr, 0,
        sizeof(int) * total_workers * profile_capacity(),
        streams[2 % NUM_STREAMS]));
    GTAP_DETAIL_CUDA_TRY(cudaMemsetAsync(
        profile_dropped_events_ptr, 0,
        sizeof(unsigned long long) * total_workers, streams[0]));
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    // Reinitialize warp ID pools metadata
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size,
        runtime_config.warps_per_block * warp_size>>>();
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    free(streams);

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

    gtap::detail::launch_config launch_config{
        config.grid_size,
        config.block_size,
        config.block_size / gtap::detail::warp_size,
        config.grid_size * (config.block_size / gtap::detail::warp_size),
        config.max_tasks_per_warp,
        config.num_queues,
        config.max_tasks_per_warp / config.num_queues,
        config.profile_capacity_per_warp,
        gtap::detail::thread::dynamic_shared_bytes(config.block_size, config.num_queues)
    };
    GTAP_DETAIL_CUDA_TRY(gtap::detail::publish_launch_config(launch_config));
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
