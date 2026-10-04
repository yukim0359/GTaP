#pragma once

#include <cuda_runtime.h>

#include "runtime_config.cuh"

namespace gtap::detail {

__device__ __forceinline__ unsigned int get_lane_id() {
    return threadIdx.x & 31;
}

__device__ __forceinline__ unsigned int get_warp_id_in_block() {
    return threadIdx.x >> 5;
}

__device__ __forceinline__ unsigned int get_warp_id_global() {
    return blockIdx.x * d_launch_config.warps_per_block + get_warp_id_in_block();
}

}  // namespace gtap::detail
