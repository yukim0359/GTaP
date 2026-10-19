#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#define GTAP_DETAIL_MAX_THREADS_PER_BLOCK 1024

// #define GTAP_DETAIL_INTERNAL_DEBUG

// Safety thresholds for error detection
#define GTAP_DETAIL_QUEUE_MARGIN 100
#define GTAP_DETAIL_TASK_ID_POOL_MIN_FREE 100

namespace gtap::detail {

struct launch_config {
    int grid_size;
    int block_size;
    int warps_per_block;
    int total_scheduling_units;
    int tasks_per_scheduling_unit;
    int num_queues;
    int queue_capacity;
    int profile_interval_capacity;
    size_t dynamic_shared_bytes;
};

inline launch_config h_launch_config{};
__constant__ launch_config d_launch_config;

inline cudaStream_t h_stream = nullptr;

enum class runtime_state {
    uninitialized,
    ready,
    failed,
};

inline runtime_state h_runtime_state = runtime_state::uninitialized;

inline cudaError_t publish_launch_config(const launch_config& config) {
    cudaError_t status =
        cudaMemcpyToSymbol(d_launch_config, &config, sizeof(config));
    if (status == cudaSuccess) {
        h_launch_config = config;
    }
    return status;
}

}  // namespace gtap::detail
