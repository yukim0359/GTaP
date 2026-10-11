#pragma once

#include <cuda_runtime.h>

#include "runtime_config.cuh"

namespace gtap::detail {

__device__ __forceinline__ int get_random_block_id(int self_block) {
    unsigned int seed = (unsigned int)(clock64() + self_block * 1234);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int total_blocks = d_launch_config.grid_size;
    int r = seed % total_blocks;
    if (r == self_block) r = (r + 1) % total_blocks;
    return r;
}

__device__ __forceinline__ int get_random_warp_id_global(int self_warp) {
    unsigned int seed = (unsigned int)(clock64() + self_warp * 2654435761u);
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    int total_warps = d_launch_config.total_scheduling_units;
    int r = seed % total_warps;
    if (r == self_warp) r = (r + 1) % total_warps;
    return r;
}

}  // namespace gtap::detail
