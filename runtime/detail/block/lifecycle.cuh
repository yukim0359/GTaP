#pragma once

// Call order for every block backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.

#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

#include "profile_buffer.cuh"
#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& c = h_launch_config;
    const size_t scheduling_units = static_cast<size_t>(c.total_scheduling_units);
    const size_t tasks = scheduling_units * c.tasks_per_scheduling_unit;
    return queue_storage_allocation_bytes(scheduling_units, tasks, c.num_queues) +
           task_pool_allocation_bytes(scheduling_units, tasks, c.block_size) +
           profile_buffer_allocation_bytes(scheduling_units);
}

cudaError_t initialize_runtime() {
    GTAP_DETAIL_CUDA_TRY(initialize_runtime_error_record());
    const launch_config& runtime_config = h_launch_config;
    const size_t total_scheduling_units = runtime_config.total_scheduling_units;
    const size_t total_tasks = total_scheduling_units * runtime_config.tasks_per_scheduling_unit;

    cudaStream_t stream = h_stream;

    queue_storage_buffers queues{};
    GTAP_DETAIL_CUDA_TRY(stage_queue_storage(
        total_scheduling_units, total_tasks, runtime_config.num_queues, stream,
        &queues));
    task_pool_buffers task_pool{};
    GTAP_DETAIL_CUDA_TRY(stage_task_pool(
        total_scheduling_units, total_tasks, runtime_config.block_size, stream,
        &task_pool));
    profile_buffers profile{};
    GTAP_DETAIL_CUDA_TRY(stage_profile_buffers(
        total_scheduling_units, stream, &profile));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(stream));

    GTAP_DETAIL_CUDA_TRY(publish_queue_storage(queues));
    GTAP_DETAIL_CUDA_TRY(publish_task_pool(task_pool));
    GTAP_DETAIL_CUDA_TRY(publish_profile_buffers(profile));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(reset_queue_counters());
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_active_block_count, &one, sizeof(int)));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1, 0, stream>>>();
    // TODO: cudaDeviceSynchronize waits for every stream.
    return cudaDeviceSynchronize();
}

cudaError_t finalize_runtime() {
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
        total_scheduling_units, total_tasks, runtime_config.num_queues, stream));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(
        total_scheduling_units, total_tasks, runtime_config.block_size, stream));
    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(
        total_scheduling_units, stream));
    GTAP_DETAIL_CUDA_TRY(cudaStreamSynchronize(stream));

    int zero = 0;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_first_task_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_all_tasks_finished, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_runtime_error_code, &zero, sizeof(int)));
    GTAP_DETAIL_CUDA_TRY(reset_queue_counters());
    int one = 1;
    GTAP_DETAIL_CUDA_TRY(cudaMemcpyToSymbol(
        d_active_block_count, &one, sizeof(int)));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1, 0, stream>>>();
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY(cudaDeviceSynchronize());
    return cudaGetLastError();
}

}  // namespace gtap::detail::block
