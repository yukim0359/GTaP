#pragma once

#include <cuda_runtime.h>

#include "runtime_config.cuh"

namespace gtap::detail {

__device__ __forceinline__ int get_random_block_id(int selfBlock) {
    unsigned int seed = (unsigned int)(clock64() + selfBlock * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int totalBlocks = d_launch_config.grid_size;
    int r = seed % totalBlocks;
    if (r == selfBlock) r = (r + 1) % totalBlocks;
    return r;
}

__device__ __forceinline__ int get_random_warp_id_global(int selfWarp) {
    unsigned int seed = (unsigned int)(clock64() + selfWarp * 2654435761u);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int totalWarps = d_launch_config.total_workers;
    int r = seed % totalWarps;
    if (r == selfWarp) r = (r + 1) % totalWarps;
    return r;
}

}  // namespace gtap::detail
