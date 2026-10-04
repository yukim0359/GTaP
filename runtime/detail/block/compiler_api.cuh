#pragma once

#include "task_ops.cuh"

__device__ inline void __gtap_execute_task_loop() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    gtap::detail::block::execute_task_loop<gtap::detail::TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    gtap::detail::block::execute_task_loop<gtap::detail::TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}

__device__ __forceinline__ int __gtap_get_task_state(int tid) {
    return gtap::detail::block::get_task_state(tid);
}

__device__ __forceinline__ bool __gtap_set_state_for_join_block(
    int tid, gtap::detail::block::TaskContext* ctx, int next_state, int unused_value
) {
    return gtap::detail::block::set_state_for_join_block(tid, ctx, next_state, unused_value);
}

__device__ void __gtap_finish_task(int tid, gtap::detail::block::TaskContext* ctx) {
    gtap::detail::block::finish_task(tid, ctx);
}

__device__ __forceinline__ void* __gtap_spawn_task(
    gtap::detail::block::TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, gtap::detail::block::TaskContext*),
    int unused_value
) {
    return gtap::detail::block::spawn_task(ctx, self_tid, child_count, func, unused_value);
}

__device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, gtap::detail::block::TaskContext*),
    int unused_value
) {
    gtap::detail::block::push_initial_task(func, unused_value);
}

__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return gtap::detail::block::get_task_data(tid);
}

__device__ __forceinline__ void* __gtap_get_entry_result_data() {
    return gtap::detail::block::get_entry_result_data();
}
