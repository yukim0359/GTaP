#pragma once

#include <cuda_runtime.h>

namespace gtap::detail::block {

struct TaskContext;

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

struct TaskContext {
    int generated_task_count;
    int queue_tail;
    int id_list_alloc_pos;
    int id_list_free_pos_stale;
    // TODO: Drop have_task_id_resumable and use task_id_resumable == -1.
    bool have_task_id_resumable;
    int task_id_resumable;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // TODO: Only generation, parent_tid, and parent_generation are read.
    // Store those fields, as thread mode does, instead of the whole TaskHeader.
    TaskHeader cached_task_header;
#endif
};

}  // namespace gtap::detail::block
