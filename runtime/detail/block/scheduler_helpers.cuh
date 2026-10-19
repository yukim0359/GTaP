#pragma once

#include "../common/cuda_primitives.cuh"

#include "profile_buffer.cuh"
#include "task_pool.cuh"

namespace gtap::detail::block {

using namespace gtap::detail;

#ifdef GTAP_ENABLE_PROFILING
__device__ __forceinline__ void record_execution_start(int* working_time_idx) {
    if (threadIdx.x == 0) {
        if (*working_time_idx + 1 < profile_timestamp_capacity()) {
            const size_t profile_idx =
                static_cast<size_t>(blockIdx.x) * profile_timestamp_capacity() +
                *working_time_idx;
            working_time[profile_idx] = get_global_time();
            (*working_time_idx)++;
        } else {
            atomicAdd(&profile_dropped_events[blockIdx.x], 1ULL);
        }
    }
}

__device__ __forceinline__ void record_execution_end(int* working_time_idx) {
    if (threadIdx.x == 0) {
        if (*working_time_idx < profile_timestamp_capacity()) {
            const size_t profile_idx =
                static_cast<size_t>(blockIdx.x) * profile_timestamp_capacity() +
                *working_time_idx;
            working_time[profile_idx] = get_global_time();
            (*working_time_idx)++;
        }
    }
}
#endif

// finish_task advances the context generation only when the task finishes.
// A taskwait suspension leaves it unchanged. The id is released after the
// execute loop's threadfence, before push replaces the executed id.
__device__ __forceinline__ void release_task_id(
    int executed_tid
#ifndef GTAP_ASSUME_NO_TASKWAIT
    , uint16_t generation_before
    , const TaskContext* task_context
#endif
) {
    if (threadIdx.x != 0) return;
#ifdef GTAP_ASSUME_NO_TASKWAIT
    release_task_id_to_block_pool(executed_tid);
#else
    if (task_context->generation != generation_before) {
        release_task_id_to_block_pool(executed_tid);
    }
#endif
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
// Copy task header to TaskContext for reuse in task function (using L2 load)
__device__ __forceinline__ void copy_task_header(
    int execute_task_id,
    TaskContext* task_context
) {
    if (threadIdx.x == 0) {
        TaskHeader* src_hdr = &d_task_headers[execute_task_id];
        task_context->parent_tid = load_L2(&src_hdr->parent_tid);
        unsigned int generations =
            load_L2(reinterpret_cast<unsigned int*>(&src_hdr->generation));
        task_context->generation = static_cast<uint16_t>(generations);
        task_context->parent_generation =
            static_cast<uint16_t>(generations >> 16);
    }
    __syncthreads();
}
#endif

}  // namespace gtap::detail::block
