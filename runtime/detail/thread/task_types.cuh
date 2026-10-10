#pragma once

#include "../common/cuda_primitives.cuh"

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
    int        parent_tid;
    int        waiting_child_count;
    // Adjacent and 4-byte aligned so both halves load as one 32-bit word.
    // Little-endian: generation is the low half, parent_generation the high half.
    uint16_t   generation;
    uint16_t   parent_generation;
    uint16_t   state;
    uint16_t   queue_idx;
#endif
};

#ifndef GTAP_ASSUME_NO_TASKWAIT
// copy_task_header loads both fields as one 32-bit word. That load is valid only
// when generation is 4-byte aligned and parent_generation is the next half.
static_assert(offsetof(TaskHeader, generation) % 4 == 0,
              "generation is 4-byte aligned");
static_assert(offsetof(TaskHeader, parent_generation) ==
                  offsetof(TaskHeader, generation) + sizeof(uint16_t),
              "parent_generation follows generation");
#endif

}  // namespace gtap::detail::thread
