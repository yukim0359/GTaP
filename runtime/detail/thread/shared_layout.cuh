#pragma once

#include "task_types.cuh"

namespace gtap::detail::thread {

using namespace gtap::detail;

// Offsets into the dynamic shared memory of one CUDA block.
struct shared_layout {
    // Arrays referenced by TaskContext fields.
    size_t generated_task_counts;
    size_t queue_tails;
    size_t staged_task_ids;
    // Cached lengths used to choose which queue DAQ pops.
    size_t queue_lengths;
    // Index used by profiling.
    size_t working_time_idx;
    // Total shared memory bytes for the block.
    size_t bytes;
};

__host__ __device__ inline shared_layout shared_layout_for(
    int warps_per_block, int num_queues, bool include_queue_tails
) {
    shared_layout layout{};
    size_t cursor = sizeof(TaskContext) * static_cast<size_t>(warps_per_block);
    cursor = align_up(cursor, alignof(int));
    const size_t block_queue_int_bytes =
        sizeof(int) * static_cast<size_t>(warps_per_block) *
        static_cast<size_t>(num_queues);
    layout.generated_task_counts = cursor;
    cursor += block_queue_int_bytes;
    layout.queue_tails = cursor;
    if (include_queue_tails) {
        cursor += block_queue_int_bytes;
    }
    layout.staged_task_ids = cursor;
    cursor += block_queue_int_bytes * warp_size;
    layout.queue_lengths = cursor;
    if (num_queues > 1) {
        cursor += block_queue_int_bytes;
    }
#ifdef GTAP_ENABLE_PROFILING
    layout.working_time_idx = cursor;
    cursor += sizeof(int) * static_cast<size_t>(warps_per_block);
#endif
    layout.bytes = cursor;
    return layout;
}

}  // namespace gtap::detail::thread
