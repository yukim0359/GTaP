#pragma once

#include <cuda_runtime.h>

#ifdef GTAP_ENABLE_PROFILING
// Implemented by each block runtime (standard / gq / ...).
cudaError_t get_working_time_data(long long* host_working_time);
cudaError_t get_block_profile_dropped_events_data(unsigned long long* host_dropped);
__global__ void get_final_working_time_indices(int* indices);
#endif
