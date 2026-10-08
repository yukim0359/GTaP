#pragma once

// Call order for every block backend.
// Queue storage, the task pool, and profile buffers are allocated by their owners.
// All clears use h_stream.
// A failed initialize releases staged buffers. The caller owns h_stream.

#include <climits>

#include "../common/runtime_config.cuh"
#include "../common/runtime_error.cuh"

#include "profile_buffer.cuh"
#include "scheduler.cuh"
#include "task_pool.cuh"
#include "termination.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

static size_t runtime_device_allocation_bytes() {
    const launch_config& config = h_launch_config;
    return queue_storage_allocation_bytes(config) +
           task_pool_allocation_bytes(config) +
           profile_buffer_allocation_bytes(config);
}

inline size_t task_management_bytes(const launch_config& config) {
    const size_t entry_result =
        __gtap_auto_entry_result_size * static_cast<size_t>(config.block_size);
    return task_pool_allocation_bytes(config) - entry_result
        + queue_storage_allocation_bytes(config);
}

// A task count of 0 means unset. A budget of 0 means unset.
// One set value is used as given. Both set takes the smaller slot count.
// Both unset uses default_tasks_per_scheduling_unit.
// The task region is fixed bytes plus a constant increment per slot.
inline int tasks_within_budget(launch_config config, size_t budget) {
    if (budget == 0) {
        return config.tasks_per_scheduling_unit == 0
            ? default_tasks_per_scheduling_unit
            : config.tasks_per_scheduling_unit;
    }

    int ceiling = config.tasks_per_scheduling_unit;
    if (ceiling == 0) {
        const int units = config.total_scheduling_units;
        if (units <= 0) return 0;
        // id = slot * units + unit must fit in a signed int.
        ceiling = INT_MAX / units;
    }
    if (ceiling <= 0) return 0;

    config.tasks_per_scheduling_unit = 0;
    config.queue_capacity = 0;
    const size_t bytes_with_no_slots = task_management_bytes(config);
    if (budget <= bytes_with_no_slots) return 0;
    config.tasks_per_scheduling_unit = 1;
    config.queue_capacity = 1;
    const size_t bytes_with_one_slot = task_management_bytes(config);
    if (bytes_with_one_slot <= bytes_with_no_slots) return 0;
    const size_t bytes_per_slot = bytes_with_one_slot - bytes_with_no_slots;
    size_t tasks = (budget - bytes_with_no_slots) / bytes_per_slot;
    if (tasks > static_cast<size_t>(ceiling)) tasks = static_cast<size_t>(ceiling);
    return static_cast<int>(tasks);
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

    cudaStream_t stream = h_stream;

    GTAP_DETAIL_CUDA_TRY_OR(
        stage_queue_storage(runtime_config, stream, &queues),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_task_pool(runtime_config, stream, &task_pool),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        stage_profile_buffers(runtime_config, stream, &profile),
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
    GTAP_DETAIL_CUDA_TRY_OR(
        reset_queue_counters(),
        abandon_initialize(&queues, &task_pool, &profile));
    int one = 1;
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaMemcpyToSymbol(d_active_block_count, &one, sizeof(int)),
        abandon_initialize(&queues, &task_pool, &profile));

    init_block_id_pools_metadata<<<runtime_config.grid_size, 1, 0, stream>>>();
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
    // TODO: cudaDeviceSynchronize waits for every stream.
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaDeviceSynchronize(),
        abandon_initialize(&queues, &task_pool, &profile));
    GTAP_DETAIL_CUDA_TRY_OR(
        cudaGetLastError(),
        abandon_initialize(&queues, &task_pool, &profile));
    return cudaSuccess;
}

cudaError_t finalize_runtime() {
    GTAP_DETAIL_CUDA_TRY(free_queue_storage());
    GTAP_DETAIL_CUDA_TRY(free_task_pool());
    GTAP_DETAIL_CUDA_TRY(free_profile_buffers());
    GTAP_DETAIL_CUDA_TRY(finalize_runtime_error_record());
    return cudaGetLastError();
}

cudaError_t reset_runtime() {
    reset_runtime_error_record();
    const launch_config& runtime_config = h_launch_config;

    cudaStream_t stream = h_stream;

    GTAP_DETAIL_CUDA_TRY(clear_queue_storage(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(clear_task_pool(runtime_config, stream));
    GTAP_DETAIL_CUDA_TRY(clear_profile_buffers(runtime_config, stream));
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
