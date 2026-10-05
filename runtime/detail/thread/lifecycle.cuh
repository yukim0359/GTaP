#pragma once

// Call order for every thread backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.
// A failed initialize releases staged buffers. The caller owns h_stream.

#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

#include "profile_buffer.cuh"
#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = h_launch_config;
    const size_t scheduling_units = static_cast<size_t>(c.total_scheduling_units);
    const size_t tasks = scheduling_units * c.tasks_per_scheduling_unit;
    return queue_storage_allocation_bytes(scheduling_units, c.num_queues) +
           task_pool_allocation_bytes(scheduling_units, tasks) +
           profile_buffer_allocation_bytes(scheduling_units);
}

inline void abandon_initialize(
    queue_storage_buffers* queues,
    task_pool_buffers* task_pool,
    profile_buffers* profile
) {
    cudaStreamSynchronize(h_stream);
    release_staged_queue_storage(queues);
    release_staged_task_pool(task_pool);
    release_staged_profile_buffers(profile);
    finalize_runtime_error_record();
    cudaGetLastError();
}

cudaError_t initialize_runtime() {
    queue_storage_buffers queues{};
    task_pool_buffers task_pool{};
    profile_buffers profile{};
    GTAP_DETAIL_CUDA_TRY_OR(
        initialize_runtime_error_record(),
        abandon_initialize(&queues, &task_pool, &profile));
    const launch_config& runtime_config = h_launch_config;
    const size_t total_scheduling_units = runtime_config.total_scheduling_units;
    const size_t total_tasks = total_scheduling_units * runtime_config.tasks_per_scheduling_unit;
    cudaStream_t stream = h_stream;

#ifdef GTAP_INTERNAL_PROFILE_INIT
    printf("\n=== initialize_runtime detailed profiling ===\n");
    // TODO: Events leak when initialize returns early.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
#endif

    GTAP_DETAIL_CUDA_TRY_OR(
        stage_queue_storage(
            total_scheduling_units, runtime_config.num_queues, stream, &queues),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_task_pool(total_scheduling_units, total_tasks, stream, &task_pool),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_profile_buffers(total_scheduling_units, stream, &profile),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaStreamSynchronize(stream),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_queue_storage(queues),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_task_pool(task_pool),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        publish_profile_buffers(profile),
        abandon_initialize(&queues, &task_pool, &profile));

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start);
#endif
    int zero = 0;
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
#endif
    // Initialize d_active_warp_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_warp_count): %.3f ms\n", elapsed);
#endif

#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(start, stream);
#endif
    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0, stream>>>();
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventRecord(stop, stream);
#endif
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaDeviceSynchronize(),
        abandon_initialize(&queues, &task_pool, &profile));
#ifdef GTAP_INTERNAL_PROFILE_INIT
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  init_warp_id_pools_metadata kernel: %.3f ms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("=== initialize_runtime profiling complete ===\n\n");
#endif

    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
    return cudaSuccess;
}

cudaError_t finalize_runtime() {
    // TODO: After gtap_synchronize has printed the GTaP record, skip the generic
    // "CUDA ERROR: unspecified launch failure" and only return the status.
    GTAP_DETAIL_CUDA_TRY(free_queue_storage());
    GTAP_DETAIL_CUDA_TRY(free_task_pool());
    GTAP_DETAIL_CUDA_TRY(free_profile_buffers());
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());
    return cudaGetLastError();
}

cudaError_t reset_runtime() {
    reset_runtime_error_record_host();
    const launch_config& runtime_config = h_launch_config;
    const size_t total_scheduling_units = runtime_config.total_scheduling_units;
    const size_t total_tasks = total_scheduling_units * runtime_config.tasks_per_scheduling_unit;
    cudaStream_t stream = h_stream;

    GTAP_DETAIL_CUDA_TRY(clear_queue_storage(
        total_scheduling_units, runtime_config.num_queues, stream));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(
        total_scheduling_units, total_tasks, stream));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(d_active_warp_count, &one, sizeof(int)));

    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(
        total_scheduling_units, stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(stream));

    init_warp_id_pools_metadata<<<
        runtime_config.grid_size, runtime_config.block_size, 0, stream>>>();
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    return cudaGetLastError();
}

}  // namespace gtap::detail::thread
