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
    int parent_tid;
    int waiting_child_count;
    // Adjacent and 4-byte aligned so both halves load as one 32-bit word.
    // Little-endian: generation is the low half, parent_generation the high half.
    uint16_t generation;
    uint16_t parent_generation;
    uint16_t state;
#endif
};

#ifndef GTAP_ASSUME_NO_TASKWAIT
// copy_task_header loads the header pair as one 32-bit word and stores that
// word onto the TaskContext pair. Both pairs must be 4-byte aligned and adjacent.
static_assert(offsetof(TaskHeader, generation) % 4 == 0,
              "generation is 4-byte aligned");
static_assert(offsetof(TaskHeader, parent_generation) ==
                  offsetof(TaskHeader, generation) + sizeof(uint16_t),
              "parent_generation follows generation");
static_assert(offsetof(TaskContext, generation) % 4 == 0,
              "TaskContext generation is 4-byte aligned");
static_assert(offsetof(TaskContext, parent_generation) ==
                  offsetof(TaskContext, generation) + sizeof(uint16_t),
              "TaskContext parent_generation follows generation");
#endif

}  // namespace gtap::detail::block
