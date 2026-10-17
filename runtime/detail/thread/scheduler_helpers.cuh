#pragma once

#include "../common/cuda_primitives.cuh"

#include "profile_buffer.cuh"
#include "task_pool.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__device__ __forceinline__ void record_execution_start(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    int execute_task_count,
    int* working_time_idx
) {
    if (lane == 0) {
        if (working_time_idx[warp_id_in_block] + 1 <
            profile_timestamp_capacity()) {
            const int profile_idx =
                warp_id_global * profile_timestamp_capacity() +
                working_time_idx[warp_id_in_block];
            working_time[profile_idx] = get_global_time();
            tasks_processed_count[profile_idx] = execute_task_count;
            working_time_idx[warp_id_in_block]++;
        } else {
            atomicAdd(&profile_dropped_events[warp_id_global], 1ULL);
        }
    }
}

__device__ __forceinline__ void record_execution_end(
    int warp_id_in_block,
    int warp_id_global,
    int lane,
    int execute_task_count,
    int* working_time_idx
) {
    if (lane == 0) {
        if (working_time_idx[warp_id_in_block] < profile_timestamp_capacity()) {
            const int profile_idx =
                warp_id_global * profile_timestamp_capacity() +
                working_time_idx[warp_id_in_block];
            working_time[profile_idx] = get_global_time();
            tasks_processed_count[profile_idx] = execute_task_count;
            working_time_idx[warp_id_in_block]++;
        }
    }
}
#endif

// finish_task advances the low half of generations[lane] only when the
// task finishes. A taskwait suspension leaves it unchanged. Every lane calls
// this after the execute loop's threadfence.
__device__ __forceinline__ void release_task_ids(
    int lane,
    int execute_task_count,
    int executed_tid,
    int alloc_pos
#ifndef GTAP_ASSUME_NO_TASKWAIT
    , uint16_t generation_before
    , const TaskContext* task_context
#endif
) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    const bool returns_id = lane < execute_task_count;
#else
    const bool returns_id =
        lane < execute_task_count &&
        static_cast<uint16_t>(task_context->generations[lane]) != generation_before;
#endif
    const unsigned mask = __ballot_sync(0xffffffffu, returns_id);
    if (mask == 0) return;

    const int leader = __ffs(mask) - 1;
    int base = 0;
    int publish_count = 0;
    if (lane == leader) {
        const int pool_size = d_launch_config.tasks_per_scheduling_unit;
        int* const free_pos_ptr =
            &d_task_id_list_free_positions[get_warp_id_global()];
        const int free_pos = load_L2(free_pos_ptr);
        const int max_safe_ticket =
            max_safe_release_ticket(alloc_pos, pool_size);
        int releasable_count = max_safe_ticket - free_pos + 1;
        if (releasable_count < 0) releasable_count = 0;
        const int id_count_to_return = __popc(mask);
        publish_count = releasable_count < id_count_to_return ? releasable_count : id_count_to_return;
        if (publish_count > 0) {
            base = atomicAdd(free_pos_ptr, publish_count);
        }
    }
    base = __shfl_sync(0xffffffffu, base, leader);
    publish_count = __shfl_sync(0xffffffffu, publish_count, leader);
    if (!returns_id) return;

    const int tids_before_this_lane = __popc(mask & ((1u << lane) - 1u));
    if (tids_before_this_lane >= publish_count) return;
    release_task_id_to_warp_pool(executed_tid, base + tids_before_this_lane);
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Copy task header to TaskContext for reuse in task function (using L2 load)
__device__ __forceinline__ void copy_task_header(
    int lane,
    int execute_task_id,
    TaskContext* task_context
) {
    TaskHeader* src_hdr = &d_task_headers[execute_task_id];
    task_context->parent_tids[lane] = load_L2(&src_hdr->parent_tid);
    task_context->generations[lane] =
        load_L2(reinterpret_cast<unsigned int*>(&src_hdr->generation));
}
#endif

}  // namespace gtap::detail::thread
