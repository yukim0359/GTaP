#pragma once

#include "../common/runtime.cuh"

namespace gtap::detail::thread {

__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished;
__device__ int d_active_warp_count;

}  // namespace gtap::detail::thread
