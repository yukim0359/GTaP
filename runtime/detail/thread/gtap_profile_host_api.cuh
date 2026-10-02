#pragma once

#include <cuda_runtime.h>

#ifdef GTAP_ENABLE_PROFILING
// Implemented by each block runtime (standard / gq / ...).
namespace gtap::detail::thread {
cudaError_t get_warp_working_time_data(long long* host_working_time);
cudaError_t get_warp_tasks_processed_count_data(int* host_counts);
cudaError_t get_warp_profile_dropped_events_data(unsigned long long* host_counts);
__global__ void get_final_warp_working_time_indices(int* indices);
}
#endif
