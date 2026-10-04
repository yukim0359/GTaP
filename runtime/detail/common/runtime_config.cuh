#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#define GTAP_MAX_THREADS_PER_BLOCK 1024

// #define GTAP_DETAIL_INTERNAL_DEBUG

// Safety thresholds for error detection
#define GTAP_DETAIL_QUEUE_MARGIN 100
#define GTAP_DETAIL_TASK_ID_POOL_MIN_FREE 100

namespace gtap::detail {

struct launch_config {
    int grid_size;
    int block_size;
    int warps_per_block;
    int total_workers;
    int tasks_per_worker;
    int num_queues;
    int queue_capacity;
    int profile_interval_capacity;
    size_t dynamic_shared_bytes;
};

__constant__ launch_config d_launch_config;

inline launch_config& stored_launch_config() {
    static launch_config config{};
    return config;
}

inline cudaStream_t& stored_stream() {
    static cudaStream_t stream = nullptr;
    return stream;
}

inline bool& initialized_flag() {
    static bool initialized = false;
    return initialized;
}

inline cudaError_t publish_launch_config(const launch_config& config) {
    stored_launch_config() = config;
    return cudaMemcpyToSymbol(d_launch_config, &config, sizeof(config));
}

}  // namespace gtap::detail
