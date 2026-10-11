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

// finish_task advances the low half of task_generations[lane] only when the
// task finishes. A taskwait suspension leaves it unchanged. The id is released
// after the execute loop's threadfence, before push replaces the executed id.
__device__ __forceinline__ void release_task_ids(
    int lane,
    int execute_task_count,
    int executed_tid
#ifndef GTAP_ASSUME_NO_TASKWAIT
    , uint16_t generation_before
    , const TaskContext* task_context
#endif
) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    release_task_id_to_warp_pool(executed_tid, lane < execute_task_count);
#else
    const bool returns_id =
        lane < execute_task_count &&
        static_cast<uint16_t>(task_context->task_generations[lane]) !=
            generation_before;
    release_task_id_to_warp_pool(executed_tid, returns_id);
#endif
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Copy task header to TaskContext for reuse in task function (using L2 load)
__device__ __forceinline__ void copy_task_header(
    int lane,
    int execute_task_id,
    TaskContext* task_context
) {
    TaskHeader* src_hdr = &d_task_headers[execute_task_id];
    task_context->task_parent_tids[lane] = load_L2(&src_hdr->parent_tid);
    task_context->task_generations[lane] =
        load_L2(reinterpret_cast<unsigned int*>(&src_hdr->generation));
}
#endif

}  // namespace gtap::detail::thread
