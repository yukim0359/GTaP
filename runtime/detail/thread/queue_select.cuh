#pragma once

#include "../common/runtime.cuh"

namespace gtap::detail::thread {

__device__ __forceinline__ int select_next_fullest_queue_idx(
    int* queue_lengths, int num_queues
) {
    int max_k = 0;
    int max_count = -1;
    for (int k = 0; k < num_queues; ++k) {
        if (queue_lengths[k] > max_count) {
            max_count = queue_lengths[k];
            max_k = k;
        }
    }
    queue_lengths[max_k] = -1;
    return max_k;
}

}  // namespace gtap::detail::thread
