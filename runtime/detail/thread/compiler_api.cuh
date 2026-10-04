#pragma once

#include "task_ops.cuh"

__device__ __forceinline__ void __gtap_execute_task_loop() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    gtap::detail::thread::execute_task_loop<gtap::detail::TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    gtap::detail::thread::execute_task_loop<gtap::detail::TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}

__device__ __forceinline__ int __gtap_get_task_state(int tid) {
    return gtap::detail::thread::get_task_state(tid);
}

__device__ __forceinline__ bool __gtap_set_state_for_join(
    int tid, int child_count, int next_state, int queue_idx_after_join
) {
    return gtap::detail::thread::set_state_for_join(
        tid, child_count, next_state, queue_idx_after_join);
}

__device__ __forceinline__ void __gtap_finish_task(
    int tid, gtap::detail::thread::TaskContext* ctx
) {
    gtap::detail::thread::finish_task(tid, ctx);
}

__device__ __forceinline__ void* __gtap_spawn_task(
    gtap::detail::thread::TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, gtap::detail::thread::TaskContext*),
    int child_queue_idx
) {
    return gtap::detail::thread::spawn_task(
        ctx, self_tid, child_count, func, child_queue_idx);
}

__device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, gtap::detail::thread::TaskContext*),
    int initial_queue_idx
) {
    gtap::detail::thread::push_initial_task(func, initial_queue_idx);
}

__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return gtap::detail::thread::get_task_data(tid);
}
