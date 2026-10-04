#pragma once

#include "runtime_config.cuh"

namespace gtap::detail {

// Buffer length in timestamps. Each interval stores a start and an end.
__host__ __device__ __forceinline__ int profile_timestamp_capacity() {
#ifdef __CUDA_ARCH__
    return 2 * d_launch_config.profile_interval_capacity;
#else
    return 2 * h_launch_config.profile_interval_capacity;
#endif
}

}  // namespace gtap::detail
