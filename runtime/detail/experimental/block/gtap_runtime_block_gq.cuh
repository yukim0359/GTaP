#pragma once

#include <cuda_runtime.h>
#include <climits>
#include "../../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_BLOCK
#define __GTAP_WORKER_IS_BLOCK
#endif

#define GTAP_EXPERIMENTAL_PROFILE_LEGACY 1

#include "../../block/gtap_block_core.cuh"

#ifndef GTAP_BLOCK_SIZE
#error "GTAP block mode requires GTAP_BLOCK_SIZE at compile time"
#endif

struct gtap_block_config {
    int grid_size = 1024;
    int max_tasks_per_block = 10000;
    int profile_capacity_per_block = 15000;
    size_t dynamic_shared_bytes = 0;
    cudaStream_t stream = nullptr;
};

inline cudaError_t gtap_validate_config(const gtap_block_config& config) {
    if (config.grid_size <= 0) return cudaErrorInvalidConfiguration;
    if (config.max_tasks_per_block <= 0) return cudaErrorInvalidValue;
#ifdef GTAP_PROFILE
    if (config.profile_capacity_per_block <= 0 ||
        config.profile_capacity_per_block > INT_MAX / 2) {
        return cudaErrorInvalidValue;
    }
#endif
    return cudaSuccess;
}

__constant__ int* d_global_task_queue;
__device__ unsigned int d_queue_head;     // Global queue head (consumer reads from here)
__device__ unsigned int d_queue_tail;     // Global queue tail (consumer-visible, committed)
__device__ unsigned int d_queue_alloc;    // Write allocation position (producers reserve here)

static size_t __gtap_runtime_device_allocation_bytes() {
    const gtap_launch_config& c = gtap_stored_launch_config();
    const size_t tasks =
        static_cast<size_t>(c.total_workers) * c.tasks_per_worker;
    const size_t global_queue_bytes = sizeof(int) * tasks;
    const size_t task_id_list_bytes =
        sizeof(TaskIdList) * c.total_workers;
    const size_t task_id_pool_bytes = sizeof(int) * tasks;
    const size_t header_bytes = sizeof(TaskHeader) * tasks;
    const size_t task_data_bytes = gtap_host_task_data_stride() * tasks;
    const size_t task_id_generated_bytes =
        sizeof(int) * c.total_workers * GTAP_MAX_CHILD_TASKS;
    size_t total = global_queue_bytes + task_id_list_bytes + header_bytes +
        task_data_bytes + task_id_generated_bytes + task_id_pool_bytes;
#ifdef GTAP_PROFILE
    total += 2 * sizeof(long long) * c.total_workers *
             (2 * c.profile_interval_capacity);
#endif
    return total;
}

#define GTAP_RUNTIME_GRID_SIZE (gtap_stored_launch_config().grid_size)
#define GTAP_RUNTIME_TOTAL_TASKS \
    (gtap_stored_launch_config().total_workers * \
     gtap_stored_launch_config().tasks_per_worker)
#define GTAP_RUNTIME_TASKS_PER_BLOCK \
    (gtap_stored_launch_config().tasks_per_worker)

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());
    const gtap_launch_config& runtime_config = gtap_stored_launch_config();
    const size_t total_tasks =
        static_cast<size_t>(runtime_config.total_workers) *
        runtime_config.tasks_per_worker;

    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    int* d_global_task_queue_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_global_task_queue_ptr),
        sizeof(int) * total_tasks));
    GTAP_CUDA_TRY(cudaMemsetAsync(
        d_global_task_queue_ptr, 0, sizeof(int) * total_tasks, streams[0]));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE));
    // Lazy initialization: set id_list to -1, will be computed on first access
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE, streams[1]));  // 0xFF = -1 for all bytes
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, streams[2]));

    // Allocate static storage for task data (type-erased as byte array)
    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = gtap_host_task_data_stride();
    size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    
    int* d_task_id_generated_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_ptr), task_id_array_size));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));

    int* d_task_id_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&d_task_id_storage_ptr),
        sizeof(int) * total_tasks));

    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_global_task_queue, &d_global_task_queue_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_CUDA_TRY(gtap_init_device_task_data_stride());
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated, &d_task_id_generated_ptr, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        d_task_id_storage, &d_task_id_storage_ptr, sizeof(int*)));
    
#ifdef GTAP_PROFILE
    const size_t profile_bytes = sizeof(long long) *
        runtime_config.total_workers * gtap_profile_capacity();
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&having_task_time_ptr), profile_bytes));
    GTAP_CUDA_TRY(cudaMalloc(
        reinterpret_cast<void**>(&working_time_ptr), profile_bytes));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        having_task_time, &having_task_time_ptr, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(
        working_time, &working_time_ptr, sizeof(working_time_ptr)));
    GTAP_CUDA_TRY(cudaMemsetAsync(having_task_time_ptr, 0, profile_bytes, streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(working_time_ptr, 0, profile_bytes, streams[1]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[1]));
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    unsigned int uzero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_head, &uzero, sizeof(unsigned int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_tail, &uzero, sizeof(unsigned int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_alloc, &uzero, sizeof(unsigned int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    init_block_id_pools_metadata<<<
        runtime_config.grid_size, 1, 0, gtap_stored_stream()>>>();
    return cudaDeviceSynchronize();
}

cudaError_t __gtap_finalize_task_runtime() {
    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    int* d_task_id_storage_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_task_id_storage_ptr, d_task_id_storage, sizeof(int*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    int* d_task_id_generated_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
#ifdef GTAP_PROFILE
    long long* having_task_time_ptr = nullptr;
    long long* working_time_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &having_task_time_ptr, having_task_time, sizeof(having_task_time_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &working_time_ptr, working_time, sizeof(working_time_ptr)));
#endif

    
    // Free global queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_global_task_queue_ptr));
    }
    
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    
    if (d_task_id_generated_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_generated_ptr));
    }
    if (d_task_id_storage_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_storage_ptr));
    }
#ifdef GTAP_PROFILE
    if (having_task_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(having_task_time_ptr));
    if (working_time_ptr != nullptr) GTAP_CUDA_TRY(cudaFree(working_time_ptr));
#endif
    
    GTAP_CUDA_TRY(gtap_finalize_runtime_error_report());

    return cudaGetLastError();
}

cudaError_t gtap_initialize(
    const gtap_block_config& config,
    size_t* device_bytes_allocated = nullptr
) {
    GTAP_CUDA_TRY(gtap_validate_config(config));
    gtap_launch_config runtime_config{
        config.grid_size,
        GTAP_BLOCK_SIZE,
        GTAP_BLOCK_SIZE / GTAP_WARP_SIZE,
        config.grid_size,
        config.max_tasks_per_block,
        1,
        config.max_tasks_per_block,
        config.profile_capacity_per_block,
        config.dynamic_shared_bytes
    };
    GTAP_CUDA_TRY(gtap_publish_launch_config(runtime_config));
    gtap_stored_stream() = config.stream;
    cudaError_t err = __gtap_init_task_runtime();
    if (err == cudaSuccess) {
        gtap_initialized_flag() = true;
        gtap_store_optional_size(
            device_bytes_allocated, __gtap_runtime_device_allocation_bytes());
    }
    return err;
}

cudaError_t gtap_initialize(size_t* device_bytes_allocated = nullptr) {
    return gtap_initialize(gtap_block_config{}, device_bytes_allocated);
}

cudaError_t gtap_finalize() {
    cudaError_t err = __gtap_finalize_task_runtime();
    if (err == cudaSuccess) gtap_initialized_flag() = false;
    return err;
}

// Reset task runtime state for re-execution
cudaError_t __gtap_reset_task_runtime() {
    gtap_reset_runtime_error_report_host();

    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    int* d_global_task_queue_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(
        &d_global_task_queue_ptr, d_global_task_queue, sizeof(int*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    int* d_task_id_generated_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));

    
    // Clear global task queue
    if (d_global_task_queue_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(
            d_global_task_queue_ptr, 0,
            sizeof(int) * GTAP_RUNTIME_TOTAL_TASKS, streams[0]));
    }
    
    // Reset task ID lists (0xFF = -1 for lazy initialization)
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_RUNTIME_GRID_SIZE, streams[1]));
    }
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_RUNTIME_TOTAL_TASKS, streams[2]));
    }
    
    // Clear task data
    size_t max_task_size = gtap_host_task_data_stride();
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * GTAP_RUNTIME_TOTAL_TASKS;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    }
    
    // Clear task ID generated array
    if (d_task_id_generated_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_RUNTIME_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
    }

    // Reset profile data if enabled
#ifdef GTAP_PROFILE
    long long* having_ptr = nullptr;
    long long* working_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&having_ptr, having_task_time, sizeof(having_ptr)));
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&working_ptr, working_time, sizeof(working_ptr)));
    const size_t profile_bytes = sizeof(long long) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity();
    GTAP_CUDA_TRY(cudaMemsetAsync(having_ptr, 0, profile_bytes, streams[0]));
    GTAP_CUDA_TRY(cudaMemsetAsync(working_ptr, 0, profile_bytes, streams[1]));
#endif
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }
    
    // Reset global state
    int zero = 0;
    unsigned int uzero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_head, &uzero, sizeof(unsigned int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_tail, &uzero, sizeof(unsigned int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_queue_alloc, &uzero, sizeof(unsigned int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    // Reinitialize block ID pools metadata
    init_block_id_pools_metadata<<<GTAP_RUNTIME_GRID_SIZE, 1>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    
    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}

#ifdef GTAP_PROFILE
cudaError_t get_having_task_time_data(long long* host_having_task_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    return cudaMemcpy(host_having_task_time, ptr, sizeof(long long) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_working_time_data(long long* host_working_time) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    return cudaMemcpy(host_working_time, ptr, sizeof(long long) *
        gtap_stored_launch_config().total_workers * gtap_profile_capacity(),
        cudaMemcpyDeviceToHost);
}

cudaError_t get_block_having_task_time_data(int block_id, long long* host_having_task_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, having_task_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(host_having_task_time,
        ptr + static_cast<size_t>(block_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

cudaError_t get_block_working_time_data(int block_id, long long* host_working_time, int max_samples) {
    long long* ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&ptr, working_time, sizeof(ptr)));
    const int count = max_samples < gtap_profile_capacity() ? max_samples : gtap_profile_capacity();
    return cudaMemcpy(host_working_time,
        ptr + static_cast<size_t>(block_id) * gtap_profile_capacity(),
        sizeof(long long) * count, cudaMemcpyDeviceToHost);
}

__global__ void get_final_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (having_task_time[blockIdx.x * gtap_profile_capacity() + i] > 0) {
                count++;
            }
        }
        indices[blockIdx.x] = count;
    }
}

__global__ void get_final_working_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < gtap_profile_capacity(); i++) {
            if (working_time[blockIdx.x * gtap_profile_capacity() + i] > 0) {
                count++;
            }
        }
        indices[blockIdx.x] = count;
    }
}
#endif

// ============================================================================
// Global Queue Operations (no steal needed - all workers pop from global queue)
// ============================================================================

// Pop from global queue - block pops a single task
template<TerminationMode M>
__device__ __forceinline__ bool pop_global_queue(int* execute_task_id, bool prev_get_task) {
    bool pop_success = false;
    unsigned int head;
    // Try to claim a slot from global queue
    while (true) {
        unsigned int old_head = load_L2(&d_queue_head);
        unsigned int tail = load_L2(&d_queue_tail);
        unsigned int available = tail - old_head;  // unsigned subtraction handles wrap-around
        
        if (available == 0) break;
        
        // CAS to claim slot
        unsigned int new_head = old_head + 1;
        if (atomicCAS(&d_queue_head, old_head, new_head) == old_head) {
            head = old_head;
            pop_success = true;
            // Increment active worker count if this worker was previously idle
            if (M == TERMINATE_ON_ALL_TASKS_FINISH && !prev_get_task) {
                atomicAdd(&d_active_worker_count, 1);
            }
            break;
        }
        // CAS failed, retry
    }

    if (pop_success) {
        int idx = head % (d_gtap_launch_config.total_workers * d_gtap_launch_config.tasks_per_worker);
        *execute_task_id = load_L2(&d_global_task_queue[idx]);
#ifdef GTAP_INTERNAL_DEBUG
        printf("pop_global: tid=%d in block %d\n", *execute_task_id, blockIdx.x);
#endif
    } else {
        *execute_task_id = -1;
    }
    return pop_success;
}

// Push to global queue
template<TerminationMode M>
__device__ __forceinline__ void push_global_queue(
    TaskContext* ctx,
    int* execute_task_id,
    bool* have_execute_task
) {
    __shared__ unsigned int base_pos;
    __shared__ int first_idx_to_push;
    __shared__ int push_cnt;
    
    int total_count = (ctx->have_task_id_resumable ? 1 : 0) + ctx->task_id_generated_count;
    
    if (total_count == 0) {
        *have_execute_task = false;
        return;
    }
    
    // Determine task to execute immediately vs push to queue
    if (threadIdx.x == 0) {
        first_idx_to_push = 0;
        if (ctx->have_task_id_resumable) {
            *execute_task_id = ctx->task_id_resumable;
            *have_execute_task = true;
        } else if (ctx->task_id_generated_count > 0) {
            *execute_task_id = get_task_id_generated(blockIdx.x, 0);
            *have_execute_task = true;
            first_idx_to_push = 1;
#ifdef GTAP_INTERNAL_DEBUG
            printf("execute_immediately: tid=%d in block %d\n", *execute_task_id, blockIdx.x);
#endif
        } else {
            *have_execute_task = false;
        }
        push_cnt = ctx->task_id_generated_count - first_idx_to_push;
    }
    __syncthreads();
    
    // Push remaining tasks to global queue
    if (push_cnt <= 0) return;
    
    // Reserve slots in global queue (allocate exclusive range)
    if (threadIdx.x == 0) {
        base_pos = atomicAdd(&d_queue_alloc, (unsigned int)push_cnt);
        // Overflow check (unsigned subtraction handles wrap-around)
        unsigned int head_val = load_L2(&d_queue_head);
        if (base_pos + (unsigned int)push_cnt - head_val > (d_gtap_launch_config.total_workers * d_gtap_launch_config.tasks_per_worker) - GTAP_QUEUE_MARGIN) {
            GTAP_RECORD_QUEUE_OVERFLOW(
                -1, 0,
                static_cast<int>(base_pos + (unsigned int)push_cnt - head_val),
                (d_gtap_launch_config.total_workers * d_gtap_launch_config.tasks_per_worker) - GTAP_QUEUE_MARGIN);
        }
    }
    __syncthreads();
    
    // Write tasks to reserved slots (parallel using block threads)
    for (int j = threadIdx.x; j < push_cnt; j += blockDim.x) {
        int tid = get_task_id_generated(blockIdx.x, first_idx_to_push + j);
        unsigned int pos = (base_pos + (unsigned int)j) % (d_gtap_launch_config.total_workers * d_gtap_launch_config.tasks_per_worker);
        store_L2(&d_global_task_queue[pos], tid);
#ifdef GTAP_INTERNAL_DEBUG
        printf("push_global: tid=%d to pos %d in block %d\n", tid, pos, blockIdx.x);
#endif
        }
    __threadfence();
    __syncthreads();
    
    // Wait for prior commits and update tail (ensures in-order visibility)
    if (threadIdx.x == 0) {
        while (load_L2(&d_queue_tail) != base_pos) {
            // spin - wait for prior pushers to commit
    }
        atomicAdd(&d_queue_tail, (unsigned int)push_cnt);
    }
    __syncthreads();
}

__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state) {
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
        hdr->state = next_state;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->waiting_child_count = child_count;
#endif
    }
}

extern "C" {
__device__ __forceinline__ int __gtap_get_task_state(int tid) {
    return load_L2_u16t(&d_task_headers[tid].state);
}

__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state, int unused_value) {
    (void)unused_value;
    __gtap_set_state_for_join(tid, child_count, next_state);
}

__device__ __forceinline__ bool __gtap_set_state_for_join_block(
    int tid,
    TaskContext* ctx,
    int next_state,
    int unused_value
) {
    (void)unused_value;
    __syncthreads();
    int child_count = ctx->task_id_generated_count;
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
        hdr->state = next_state;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->waiting_child_count = child_count;
#endif
    }
    __syncthreads();
    return child_count != 0;
}

}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    __threadfence();
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
    if (rem == 1) {
        ctx->have_task_id_resumable = true;
        ctx->task_id_resumable = parentId;
    }
#ifdef GTAP_INTERNAL_DEBUG
    printf("notify_parent: %d, rem: %d\n", parentId, rem);
#endif
    return rem;
}
#endif

__device__ void __gtap_finish_task(int tid, TaskContext* ctx) {
    if (threadIdx.x == 0) {
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;
        
        if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
#ifdef GTAP_INTERNAL_DEBUG
            printf("finish_task: %d, parent_tid: %d\n", tid, parent_tid);
#endif
            notify_parent(parent_tid, ctx);
            release_task_id_to_block_pool(tid);
#else
            // NO_TASKWAIT: no need to notify parent or release child IDs
            release_task_id_to_block_pool(tid);
#endif
        } else {
            release_task_id_to_block_pool(tid);
        }
        if (tid == 0) store_L2(&d_first_task_finished, 1);
    }
}

__device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*)
) {
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    int new_tid = get_task_id_from_block_pool(
        tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->waiting_child_count = 0;
#endif
    
    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
    (void)child_count;
    return __gtap_get_task_data(new_tid);
}

extern "C" __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    return __gtap_spawn_task(ctx, self_tid, child_count, func);
}

__device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size
) {
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    int new_tid = get_task_id_from_block_pool(
        tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->waiting_child_count = 0;
#endif
    
    void* dest_task = __gtap_get_task_data(new_tid);
    memcpy(dest_task, task_data_ptr, task_data_size);

    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
    (void)child_count;
}

extern "C" __device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size,
    int unused_value
) {
    (void)unused_value;
    __gtap_spawn_task_raw(ctx, self_tid, child_count, func, task_data_ptr, task_data_size);
}

__device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*)
) { 
    TaskHeader* initial_hdr = &d_task_headers[0];
    initial_hdr->func = func;
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->waiting_child_count = 0;
#endif

    // Push to global queue (only block 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        store_L2(&d_global_task_queue[0], 0);
        __threadfence();
        store_L2(&d_queue_head, 0u);
        store_L2(&d_queue_alloc, 1u);
        store_L2(&d_queue_tail, 1u);
        __threadfence();
    }
}

extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    (void)unused_value;
    __gtap_push_initial_task(func);
}


template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    __shared__ int execute_task_id;
    __shared__ bool have_execute_task;
    __shared__ bool prev_get_task;
    __shared__ bool should_continue;
    __shared__ TaskContext block_ctx;
#ifdef GTAP_PROFILE
    __shared__ int having_task_time_idx;
    __shared__ int working_time_idx;
#endif

    if (threadIdx.x == 0) {
        should_continue = true;
        have_execute_task = false;
        block_ctx.have_task_id_resumable = false;
        block_ctx.task_id_generated_count = 0;
        block_ctx.id_list_free_pos_stale = d_gtap_launch_config.tasks_per_worker;
#ifdef GTAP_PROFILE
        working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            block_ctx.id_list_alloc_pos = 1;
            prev_get_task = true;
#ifdef GTAP_PROFILE
            having_task_time_idx = 1;
            having_task_time[blockIdx.x * gtap_profile_capacity()] = get_global_time();
#endif
        } else {
            block_ctx.id_list_alloc_pos = 0;
            prev_get_task = false;
#ifdef GTAP_PROFILE
            having_task_time_idx = 0;
#endif
        }
    }
    __syncthreads();
    
    while (should_continue) {
        if (threadIdx.x == 0) {
            if (!have_execute_task) {
                // Try to pop from global queue
                have_execute_task = pop_global_queue<M>(&execute_task_id, prev_get_task);
            }
        }
        __syncthreads();

        if (!have_execute_task) {
            if (threadIdx.x == 0) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            // Check if queue is empty (unsigned comparison handles wrap-around)
                            bool all_tasks_finished = 1;
                            unsigned int head = load_L2(&d_queue_head);
                            unsigned int tail = load_L2(&d_queue_tail);
                            if (tail - head > 0) {  // unsigned subtraction
                                all_tasks_finished = 0;
                            }
                            atomicExch(&d_all_tasks_finished_flag, all_tasks_finished);
                        }
                    }
                }
#ifdef GTAP_PROFILE
                if (prev_get_task && having_task_time_idx < gtap_profile_capacity()) {
                    having_task_time[blockIdx.x * gtap_profile_capacity() + having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = false;
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    should_continue = (load_L2(&d_all_tasks_finished_flag) == 0);
                } else {
                    should_continue = (load_L2(&d_first_task_finished) == 0);
                }
            }
            __syncthreads();
            continue;
        } else {
            if (threadIdx.x == 0) {
#ifdef GTAP_PROFILE
                // Record task start time
                if (!prev_get_task && having_task_time_idx < gtap_profile_capacity()) {
                    having_task_time[blockIdx.x * gtap_profile_capacity() + having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = true;
                block_ctx.task_id_generated_count = 0;
                block_ctx.have_task_id_resumable = false;
            }
            __syncthreads();
        }

        if (have_execute_task) {
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            if (threadIdx.x == 0) {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &block_ctx.cached_task_header;
                dst_hdr->generation = load_L2_u16t(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2_u16t(&src_hdr->parent_generation);
            }
            __syncthreads();
            
#ifdef GTAP_PROFILE
            if (threadIdx.x == 0) {
                if (working_time_idx < gtap_profile_capacity()) {
                    working_time[blockIdx.x * gtap_profile_capacity() + working_time_idx] = get_global_time();
                    working_time_idx++;
                }
            }
#endif
            void* task_data = __gtap_get_task_data(execute_task_id);
            // Read function pointer atomically (64-bit) via L2 cache
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &block_ctx);
            __threadfence();
        }
        __syncthreads();
#ifdef GTAP_PROFILE
        if (threadIdx.x == 0) {
            if (working_time_idx < gtap_profile_capacity()) {
                working_time[blockIdx.x * gtap_profile_capacity() + working_time_idx] = get_global_time();
                working_time_idx++;
            }
        }
#endif

        push_global_queue<M>(&block_ctx, &execute_task_id, &have_execute_task);
    }
#ifdef GTAP_INTERNAL_DEBUG
    if (threadIdx.x == 0) printf("execute_task_loop: end (block_id = %d)\n", blockIdx.x);
#endif
}

// Non-template device-side wrapper
extern "C" __device__ inline void __gtap_execute_task_loop_device() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}
