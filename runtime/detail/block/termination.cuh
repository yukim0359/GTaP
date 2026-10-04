#pragma once

#include <cuda_runtime.h>

namespace gtap::detail::block {

__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished;
__device__ int d_active_block_count;

}  // namespace gtap::detail::block
