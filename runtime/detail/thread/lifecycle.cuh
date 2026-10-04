#pragma once

// Standard thread backend. Queue storage stays in this file.
// Task pool and profile buffers are allocated by their owners.

#include "scheduler.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = stored_launch_config();
    const size_t workers = static_cast<size_t>(c.total_workers);
    const size_t tasks = workers * c.tasks_per_worker;
    const size_t queue_ptr_array_bytes = sizeof(WarpTaskQueueMetadata*) * c.num_queues;
    const size_t queue_metadata_bytes =
        static_cast<size_t>(c.num_queues) * sizeof(WarpTaskQueueMetadata) * workers;
    const size_t queue_storage_bytes = sizeof(int) * tasks;
    return queue_ptr_array_bytes + queue_metadata_bytes + queue_storage_bytes +
           task_pool_allocation_bytes(workers, tasks) +
           profile_buffer_allocation_bytes(workers);
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
    #endif

    task_pool_buffers task_pool{};
    GTAP_DETAIL_CUDA_TRY(stage_task_pool(
        total_workers, total_tasks,
        streams[runtime_config.num_queues],
        streams[runtime_config.num_queues + 1],
        streams[runtime_config.num_queues + 2],
        &task_pool));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queue_metadata, &d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata**)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_warp_task_queue_storage, &d_warp_task_queue_storage_ptr, sizeof(int*)));
    GTAP_DETAIL_CUDA_TRY(publish_task_pool(task_pool));
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

    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
    #endif
    GTAP_DETAIL_CUDA_TRY(allocate_profile_buffers(
        total_workers,
        streams[1 % NUM_STREAMS],
        streams[2 % NUM_STREAMS],
        streams[0]));
    #ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
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
    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * num_queues));
    if (d_warp_task_queue_metadata_ptrptr != nullptr) {
        GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * num_queues, cudaMemcpyDeviceToHost));

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

    GTAP_DETAIL_CUDA_TRY(free_task_pool());
    GTAP_DETAIL_CUDA_TRY(free_profile_buffers());
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());

    return cudaGetLastError();
}

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

    WarpTaskQueueMetadata** d_warp_task_queue_metadata_ptrptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queue_metadata_ptrptr, d_warp_task_queue_metadata, sizeof(WarpTaskQueueMetadata**)));
    int* d_warp_task_queue_storage_ptr = nullptr;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_warp_task_queue_storage_ptr, d_warp_task_queue_storage, sizeof(int*)));

    WarpTaskQueueMetadata** h_warp_task_queue_metadata_planes = reinterpret_cast<WarpTaskQueueMetadata**>(malloc(sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpy(h_warp_task_queue_metadata_planes, d_warp_task_queue_metadata_ptrptr, sizeof(WarpTaskQueueMetadata*) * runtime_config.num_queues, cudaMemcpyDeviceToHost));

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

    GTAP_DETAIL_CUDA_TRY(clear_task_pool(
        total_workers, total_tasks,
        streams[runtime_config.num_queues],
        streams[runtime_config.num_queues + 1],
        streams[runtime_config.num_queues + 2]));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));

    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(
        total_workers,
        streams[1 % NUM_STREAMS],
        streams[2 % NUM_STREAMS],
        streams[0]));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

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
