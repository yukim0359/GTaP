#pragma once

#include <cuda_runtime.h>

namespace gtap::detail::block {

struct TaskContext {
    int generated_task_count;
    int queue_tail;
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
    int task_id_resumable; // -1 when empty. Task id 0 is the root.
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int parent_tid;
    uint16_t generation;
    uint16_t parent_generation;
#endif
};

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* ctx);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // Info of current task
    uint16_t  generation;
    uint16_t  state;
    // Info of parent task
    int       parent_tid;
    uint16_t  parent_generation;
    // Info of child tasks
    int       waiting_child_count;
#endif
};

}  // namespace gtap::detail::block
