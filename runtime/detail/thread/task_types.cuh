#pragma once

#include "../common/runtime.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

struct TaskContext {
    int* generated_task_counts; // int[num_queues] for each warp
    int* queue_tails;           // int[num_queues] for each warp
    int* staged_task_ids;       // int[num_queues * warp_size] for each warp
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
    // TODO: Task functions do not read queue_idx. Move it to its own per-warp shared slot.
    int queue_idx;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int task_parent_tids[warp_size];
    uint32_t task_generations[warp_size];
#endif
};

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* __ctx);
#ifdef GTAP_ASSUME_NO_TASKWAIT
    uint16_t   queue_idx;
#else
    // Info of current task
    uint16_t   generation;
    uint16_t   state;
    uint16_t   queue_idx;
    // Info of parent task
    int        parent_tid;
    uint16_t   parent_generation;
    // Info of child tasks
    int        waiting_child_count;
#endif
};

}  // namespace gtap::detail::thread
