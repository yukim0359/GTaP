#pragma once

#include <cuda_runtime.h>
#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_BLOCK
#define __GTAP_WORKER_IS_BLOCK
#endif

#ifndef GTAP_MAX_TASKS_PER_BLOCK
#define GTAP_MAX_TASKS_PER_BLOCK 10000
#endif

#define QUEUE_SIZE (GTAP_MAX_TASKS_PER_BLOCK)

#define MAX_TASKS_GLOBAL (GTAP_MAX_TASKS_PER_BLOCK * GTAP_GRID_SIZE)

inline constexpr size_t __gtap_max_task_size = GTAP_MAX_TASK_DATA_SIZE;

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* ctx);  // function pointer
#ifndef GTAP_ASSUME_NO_TASKWAIT
    // Info of current task
    uint16_t   generation;
    uint16_t   state;
    // Info of parent task
    int   parent_tid;
    uint16_t   parent_generation;
    // Info of child tasks
    int   total_child_count;
    int   waiting_child_count;
    int   child_ids[GTAP_MAX_CHILD_TASKS];
#endif
};

// Grouped context passed to task functions to reduce parameter clutter
struct TaskContext {
    int task_id_generated_count;
    int id_list_alloc_pos;  // position to allocate next ID from
    int id_list_free_pos_stale;  // stale copy of free_pos to avoid L2 load on every allocation
#ifndef GTAP_ASSUME_NO_TASKWAIT
    bool have_task_id_resumable;
    int task_id_resumable;
    TaskHeader cached_task_header;  // cached task header for reuse in task function
#endif
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    // Shared child_ids array for the block (only when using shared memory)
    int shared_child_ids[GTAP_MAX_CHILD_TASKS + 1];
#endif
};

struct BlockTaskQueue {
    int queue[QUEUE_SIZE];
    int top;
    int bottom;
};

struct TaskIdList {
    int id_list[GTAP_MAX_TASKS_PER_BLOCK];
    int id_list_free_pos;
};

// Exposed device globals
// Note: d_task_data_bytes is now char* (byte array) to support type-erased task data (static allocation)
__device__ BlockTaskQueue* d_block_task_queues;
__device__ TaskIdList* d_task_id_lists;
__device__ TaskHeader* d_task_headers;
__device__ char* d_task_data_bytes;  // Type-erased: byte array storing task data statically
__device__ int* d_task_id_generated;
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef PROFILE
__device__ long long having_task_time[GTAP_GRID_SIZE][MAX_PROFILE_DATA];
__device__ long long working_time[GTAP_GRID_SIZE][MAX_PROFILE_DATA];
#endif

// Helper functions to access d_task_id_generated
__device__ __forceinline__ int get_task_id_generated(int block_id, int idx) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    return d_task_id_generated[offset];
}

__device__ __forceinline__ void set_task_id_generated(int block_id, int idx, int task_id) {
    int offset = block_id * GTAP_MAX_CHILD_TASKS + idx;
    d_task_id_generated[offset] = task_id;
}

// Initialize only the free_pos fields (id_list is lazily initialized on first access)
__global__ void init_block_id_pools_metadata() {
    if (threadIdx.x == 0) {
        TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
        tid_list->id_list_free_pos = GTAP_MAX_TASKS_PER_BLOCK;
    }
    __threadfence();
}

// Return value: tid and whether this slot is used for the first time (true => header already zeroed by init/reset, can skip zeroing total_child_count / waiting_child_count)
struct TaskIdFromPool {
    int tid;
    bool first_use;
};

__device__ __forceinline__ TaskIdFromPool get_task_id_from_block_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int idx = old_alloc % GTAP_MAX_TASKS_PER_BLOCK;
    int block_id = (tid_list - d_task_id_lists);  // Compute once, used in both branches
    int id;
    bool first_use = (old_alloc < GTAP_MAX_TASKS_PER_BLOCK);
    if (first_use) {
        // First pass: compute new ID directly (no L2 load needed)
        id = block_id * GTAP_MAX_TASKS_PER_BLOCK + idx;
    } else {
        // Subsequent passes: reuse released ID from id_list
        id = load_L2(&tid_list->id_list[idx]);
    }
    int free_count = *id_list_free_pos_stale - old_alloc;
    if (free_count < TASK_ID_POOL_MIN_FREE) {
        // Stale check failed, load actual value
        int new_free_pos = load_L2(&tid_list->id_list_free_pos);
        *id_list_free_pos_stale = new_free_pos;
        free_count = new_free_pos - old_alloc;
        if (free_count < TASK_ID_POOL_MIN_FREE) {
            atomicExch(&d_runtime_error_code, GTAP_ERROR_TASK_ID_POOL_EXHAUSTED);
            __trap();
        }
    }
    return TaskIdFromPool{id, first_use};
}

__device__ __forceinline__ void release_task_id_to_block_pool(int id) {
    int block_id = id / GTAP_MAX_TASKS_PER_BLOCK;
    TaskIdList* tid_list = &d_task_id_lists[block_id];
    int old_free = atomicAdd(&tid_list->id_list_free_pos, 1);
    store_L2(&tid_list->id_list[old_free % GTAP_MAX_TASKS_PER_BLOCK], id);
}

// Helper function to get task data pointer (type-erased byte array)
__device__ __forceinline__ void* __gtap_get_task_data(int tid) {
    return d_task_data_bytes + (size_t)tid * (size_t)GTAP_MAX_TASK_DATA_SIZE;
}

template <typename TaskType>
__device__ __forceinline__ TaskType* __gtap_get_task_data(int tid) {
    return reinterpret_cast<TaskType*>(__gtap_get_task_data(tid));
}

cudaError_t __gtap_init_task_runtime() {
    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_block_task_queues_ptr), sizeof(BlockTaskQueue) * GTAP_GRID_SIZE));
    CUDA_TRY(cudaMemsetAsync(d_block_task_queues_ptr, 0, sizeof(BlockTaskQueue) * GTAP_GRID_SIZE, streams[0]));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * GTAP_GRID_SIZE));
    // Lazy initialization: set id_list to -1, will be computed on first access
    CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE, streams[1]));  // 0xFF = -1 for all bytes
    
    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * MAX_TASKS_GLOBAL));
    CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * MAX_TASKS_GLOBAL, streams[2]));

    // Allocate static storage for task data (type-erased as byte array)
    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = GTAP_MAX_TASK_DATA_SIZE;
    size_t task_data_size = max_task_size * MAX_TASKS_GLOBAL;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    int* d_task_id_generated_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_ptr), task_id_array_size));
    CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
#endif
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    CUDA_TRY(cudaMemcpyToSymbol(d_block_task_queues, &d_block_task_queues_ptr, sizeof(BlockTaskQueue*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated, &d_task_id_generated_ptr, sizeof(int*)));
#endif
    
#ifdef PROFILE
    CUDA_TRY(cudaMemsetAsync(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA, streams[0]));
    CUDA_TRY(cudaMemsetAsync(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA, streams[1]));
    CUDA_TRY(cudaStreamSynchronize(streams[0]));
    CUDA_TRY(cudaStreamSynchronize(streams[1]));
#endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    int zero = 0;
    CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    init_block_id_pools_metadata<<<GTAP_GRID_SIZE, 1>>>();
    return cudaDeviceSynchronize();
}

cudaError_t __gtap_finalize_task_runtime() {
    // Get device pointers from symbols
    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queues_ptr, d_block_task_queues, sizeof(BlockTaskQueue*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    int* d_task_id_generated_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
#endif
    
    // Free allocated memory
    if (d_block_task_queues_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_block_task_queues_ptr));
    }
    
    if (d_task_id_lists_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    
    if (d_task_headers_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    if (d_task_id_generated_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_id_generated_ptr));
    }
#endif
    
    return cudaGetLastError();
}

cudaError_t gtap_initialize() {
    return __gtap_init_task_runtime();
}

cudaError_t gtap_finalize() {
    return __gtap_finalize_task_runtime();
}

// Reset task runtime state for re-execution
// This function clears all runtime state without reallocating memory
// Call this before each execution after the initial init_task_runtime call
cudaError_t __gtap_reset_task_runtime() {
    constexpr int NUM_STREAMS = 5;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    BlockTaskQueue* d_block_task_queues_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_block_task_queues_ptr, d_block_task_queues, sizeof(BlockTaskQueue*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    int* d_task_id_generated_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
#endif
    
    // Clear task queues
    if (d_block_task_queues_ptr != nullptr) {
        CUDA_TRY(cudaMemsetAsync(d_block_task_queues_ptr, 0, sizeof(BlockTaskQueue) * GTAP_GRID_SIZE, streams[0]));
    }
    
    // Reset task ID lists (0xFF = -1 for lazy initialization)
    if (d_task_id_lists_ptr != nullptr) {
        CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE, streams[1]));
    }
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * MAX_TASKS_GLOBAL, streams[2]));
    }
    
    // Clear task data
    size_t max_task_size = GTAP_MAX_TASK_DATA_SIZE;
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * MAX_TASKS_GLOBAL;
        CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[3]));
    }
    
    // Clear task ID generated array
#if !(GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    if (d_task_id_generated_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
        CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
    }
#endif
    
    // Reset profile data if enabled
#ifdef PROFILE
    CUDA_TRY(cudaMemsetAsync(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA, streams[0]));
    CUDA_TRY(cudaMemsetAsync(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA, streams[1]));
#endif
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }
    
    // Reset global state
    int zero = 0;
    CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    
    // Reinitialize block ID pools metadata
    init_block_id_pools_metadata<<<GTAP_GRID_SIZE, 1>>>();
    CUDA_TRY(cudaDeviceSynchronize());
    
    // Clean up streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    
    return cudaGetLastError();
}

cudaError_t gtap_reset() {
    return __gtap_reset_task_runtime();
}

#ifdef PROFILE
cudaError_t get_having_task_time_data(long long* host_having_task_time) {
    return cudaMemcpyFromSymbol(host_having_task_time, having_task_time, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
}

cudaError_t get_working_time_data(long long* host_working_time) {
    return cudaMemcpyFromSymbol(host_working_time, working_time, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
}

cudaError_t get_block_having_task_time_data(int block_id, long long* host_having_task_time, int max_samples) {
    long long* temp_data = (long long*)malloc(sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
    if (!temp_data) return cudaErrorMemoryAllocation;
    
    cudaError_t st = cudaMemcpyFromSymbol(temp_data, having_task_time, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
    if (st != cudaSuccess) {
        free(temp_data);
        return st;
    }
    
    for (int i = 0; i < max_samples && i < MAX_PROFILE_DATA; i++) {
        host_having_task_time[i] = temp_data[block_id * MAX_PROFILE_DATA + i];
    }
    
    free(temp_data);
    return cudaSuccess;
}

cudaError_t get_block_working_time_data(int block_id, long long* host_working_time, int max_samples) {
    long long* temp_data = (long long*)malloc(sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
    if (!temp_data) return cudaErrorMemoryAllocation;
    
    cudaError_t st = cudaMemcpyFromSymbol(temp_data, working_time, sizeof(long long) * GTAP_GRID_SIZE * MAX_PROFILE_DATA);
    if (st != cudaSuccess) {
        free(temp_data);
        return st;
    }
    
    for (int i = 0; i < max_samples && i < MAX_PROFILE_DATA; i++) {
        host_working_time[i] = temp_data[block_id * MAX_PROFILE_DATA + i];
    }
    
    free(temp_data);
    return cudaSuccess;
}

__global__ void get_final_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        // Count actual recorded samples for this block
        int count = 0;
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (having_task_time[blockIdx.x][i] > 0) {
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
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (working_time[blockIdx.x][i] > 0) {
                count++;
            }
        }
        indices[blockIdx.x] = count;
    }
}
#endif

// Chase-Lev pop: owner pops from bottom
__device__ __forceinline__ int pop(int* taskId) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    
    int b = myQueue->bottom - 1;
    store_L2(&myQueue->bottom, b);
    __threadfence();
    
    int t = load_L2(&myQueue->top);
    int size = b - t;
    
    if (size < 0) {
        store_L2(&myQueue->bottom, t);
        *taskId = -1;
        return false;
    }
    
    int task_id = load_L2(&myQueue->queue[b % QUEUE_SIZE]);
    
    if (size > 0) {
        *taskId = task_id;
#ifdef DEBUG
        printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
        return true;
    }
    
    if (atomicCAS(&myQueue->top, t, t + 1) != t) {
        *taskId = -1;
        store_L2(&myQueue->bottom, t + 1);
        return false;
    }
    
    *taskId = task_id;
    store_L2(&myQueue->bottom, t + 1);
#ifdef DEBUG
    printf("pop: %d (block: %d)\n", task_id, blockIdx.x);
#endif
    return true;
}

template<TerminationMode M>
__device__ __forceinline__ int steal(int* taskId, bool prev_get_task) {
    int targetBlock = get_random_blocknum(blockIdx.x);
    BlockTaskQueue* targetBq = &d_block_task_queues[targetBlock];
    
    int t = load_L2(&targetBq->top);
    __threadfence();
    int b = load_L2(&targetBq->bottom);
    
    int size = b - t;
    if (size <= 0) {
        *taskId = -1;
        return false;
    }
    
    int task_id = load_L2(&targetBq->queue[t % QUEUE_SIZE]);
    
    if (atomicCAS(&targetBq->top, t, t + 1) != t) {
        *taskId = -1;
        return false;
    }
    
    if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
        if (!prev_get_task) atomicAdd(&d_active_worker_count, 1);
    }
    
    *taskId = task_id;
#ifdef DEBUG
    printf("steal: %d (block: %d -> %d)\n", task_id, targetBlock, blockIdx.x);
#endif
    return true;
}

// Chase-Lev push: owner pushes to bottom
__device__ __forceinline__ void push(
    TaskContext* ctx,
    int push_total,
    int* execute_task_id
) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    
    int old_bottom = load_L2(&myQueue->bottom);
    if (threadIdx.x == 0) {
        int old_top = load_L2(&myQueue->top);
        if (old_bottom + push_total - old_top > QUEUE_SIZE - QUEUE_MARGIN) {
            atomicExch(&d_runtime_error_code, GTAP_ERROR_QUEUE_OVERFLOW);
            __trap();
        }
    }

#ifdef GTAP_ASSUME_NO_TASKWAIT
    // NO_TASKWAIT: no parent task to resume; first generated task (idx 0) is executed next.
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    *execute_task_id = ctx->shared_child_ids[0];
#else
    *execute_task_id = get_task_id_generated(blockIdx.x, 0);
#endif
    for (int i = threadIdx.x + 1; i < ctx->task_id_generated_count; i += blockDim.x) {
        int queue_idx = (old_bottom + i - 1) % QUEUE_SIZE;
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
        int val = ctx->shared_child_ids[i];
#else
        int val = get_task_id_generated(blockIdx.x, i);
#endif
        store_L2(&myQueue->queue[queue_idx], val);
    }
#else
    if (ctx->have_task_id_resumable) {
        *execute_task_id = ctx->task_id_resumable;
        for (int i = threadIdx.x; i < ctx->task_id_generated_count; i += blockDim.x) {
            int queue_idx = (old_bottom + i) % QUEUE_SIZE;
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
            int val = ctx->shared_child_ids[i];
#else
            int val = get_task_id_generated(blockIdx.x, i);
#endif
            store_L2(&myQueue->queue[queue_idx], val);
#ifdef DEBUG
            printf("push: %d (block: %d)\n", val, blockIdx.x);
#endif
        }
    } else {
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
        *execute_task_id = ctx->shared_child_ids[0];
#else
        *execute_task_id = get_task_id_generated(blockIdx.x, 0);
#endif
        for (int i = threadIdx.x + 1; i < ctx->task_id_generated_count; i += blockDim.x) {
            int queue_idx = (old_bottom + i - 1) % QUEUE_SIZE;
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
            int val = ctx->shared_child_ids[i];
#else
            int val = get_task_id_generated(blockIdx.x, i);
#endif
            store_L2(&myQueue->queue[queue_idx], val);
        }
    }
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        store_L2(&myQueue->bottom, old_bottom + push_total);
    }
}


__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state) {
    __syncthreads();
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->state = next_state;
        hdr->total_child_count = load_L2(&hdr->total_child_count) + child_count;
        // NOTE: total_child_count can be updated during the execution of task function, so we cannot use cached value here
        hdr->waiting_child_count = child_count;
#endif
    }
}

__device__ __forceinline__ int __gtap_get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2_u16t(&d_task_headers[tid].state);
#endif
}

__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state, int unused_value) {
    __gtap_set_state_for_join(tid, child_count, next_state);
}

__device__ __forceinline__ int __gtap_get_child_task_id(int parent_tid, int child_index) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    return 0;
#else
    return load_L2(&d_task_headers[parent_tid].child_ids[child_index]);
#endif
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
    if (rem == 1) {
        ctx->have_task_id_resumable = true;
        ctx->task_id_resumable = parentId;
    }
#ifdef DEBUG
    printf("notify_parent: %d, rem: %d\n", parentId, rem);
#endif
    return rem;
}
#endif

__device__ void __gtap_finish_task(int tid, TaskContext* ctx) {
    __syncthreads();
    if (threadIdx.x == 0) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
        release_task_id_to_block_pool(tid);
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#else
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;
        
        if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifdef DEBUG
            printf("finish_task: %d, parent_tid: %d, child_count: %d\n", tid, parent_tid, child_count);
#endif
            notify_parent(parent_tid, ctx);
            // If child tasks are joined, release the task IDs of child tasks
            int child_count = cached_hdr->total_child_count;
            int* child_ids = d_task_headers[tid].child_ids;
            for (int i = 0; i < child_count; i++) {
                release_task_id_to_block_pool(load_L2(&child_ids[i]));
            }
            // If the task is not waited by the parent, release the task ID of the task itself
            // TODO: Is this really necessary?
            // if (parent_waiting_child_count_old <= 0) {
            //     release_task_id_to_block_pool(tid);
            // }
        } else {
            release_task_id_to_block_pool(tid);
        }
        if (tid == 0) store_L2(&d_first_task_finished, 1);
#endif
    }
}

__device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*)
) {
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    TaskIdFromPool from_pool = get_task_id_from_block_pool(tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    int new_tid = from_pool.tid;

    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
    if (!from_pool.first_use) {
        new_hdr->state = 0;
        new_hdr->total_child_count = 0;
        new_hdr->waiting_child_count = 0;
    }
#endif
    
    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
#if (GTAP_MAX_CHILD_TASKS <= GTAP_MAX_CHILD_TASKS_FOR_SHARED)
    ctx->shared_child_ids[gen_idx] = new_tid;
#else
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
#endif
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int child_count_old = atomicAdd(child_count, 1);
    d_task_headers[self_tid].child_ids[cached_hdr->total_child_count + child_count_old] = new_tid;
#endif
    return __gtap_get_task_data(new_tid);
}

extern "C" __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int unused_value
) {
    return __gtap_spawn_task(ctx, self_tid, child_count, func);
}

__device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*)
) { 
    TaskHeader* initial_hdr = &d_task_headers[0];
    initial_hdr->func = func;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->total_child_count = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)
    
    BlockTaskQueue* bq = &d_block_task_queues[blockIdx.x];
    store_L2(&bq->queue[0], 0);
    __threadfence();
    store_L2(&bq->bottom, 1);
}

extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int unused_value
) { __gtap_push_initial_task(func); }


template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    __shared__ int execute_task_id;
    __shared__ bool have_execute_task;
    __shared__ bool prev_get_task;
    __shared__ bool should_continue;
    __shared__ TaskContext block_ctx;
#ifdef PROFILE
    __shared__ int having_task_time_idx;
    __shared__ int working_time_idx;
#endif

    if (threadIdx.x == 0) {
        should_continue = true;
        have_execute_task = false;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        block_ctx.have_task_id_resumable = false;
#endif
        block_ctx.task_id_generated_count = 0;
        block_ctx.id_list_free_pos_stale = GTAP_MAX_TASKS_PER_BLOCK;
#ifdef PROFILE
        working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            block_ctx.id_list_alloc_pos = 1;
            prev_get_task = true;
#ifdef PROFILE
            having_task_time_idx = 1;
            having_task_time[blockIdx.x][0] = get_global_time();
#endif
        } else {
            block_ctx.id_list_alloc_pos = 0;
            prev_get_task = false;
#ifdef PROFILE
            having_task_time_idx = 0;
#endif
        }
    }
    __syncthreads();
    
    while (should_continue) {
        if (threadIdx.x == 0) {
            if (!have_execute_task) {
                if (prev_get_task) {
                    have_execute_task = pop(&execute_task_id);
                }
            }
            if (!have_execute_task) {
                have_execute_task = steal<M>(&execute_task_id, prev_get_task);
            }
        }
        __syncthreads();

        if (!have_execute_task) {
            if (threadIdx.x == 0) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            bool all_tasks_finished = 1;
                            BlockTaskQueue* q = &d_block_task_queues[blockIdx.x];
                            int t = load_L2(&q->top);
                            int b = load_L2(&q->bottom);
                            if (t < b) {
                                all_tasks_finished = 0;
                            }
                            atomicExch(&d_all_tasks_finished_flag, all_tasks_finished);
                        }
                    }
                }
#ifdef PROFILE
                if (prev_get_task && having_task_time_idx < MAX_PROFILE_DATA) {
                    having_task_time[blockIdx.x][having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = false;
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    should_continue = (load_L2(&d_all_tasks_finished_flag) == 0);
                    // if (active_worker_count == 0) consecutive_idle_count++;
                    // else consecutive_idle_count = 0;
                    // should_continue = (consecutive_idle_count != NUMBER_OF_CONSECUTIVE_IDLE_COUNTS_TO_TERMINATE);
                } else {
                    should_continue = (load_L2(&d_first_task_finished) == 0);
                }
            }
            __syncthreads();
            continue;
        } else {
            if (threadIdx.x == 0) {
#ifdef PROFILE
                // Record task start time
                if (!prev_get_task && having_task_time_idx < MAX_PROFILE_DATA) {
                    having_task_time[blockIdx.x][having_task_time_idx] = get_global_time();
                    having_task_time_idx++;
                }
#endif
                prev_get_task = true;
                block_ctx.task_id_generated_count = 0;
#ifndef GTAP_ASSUME_NO_TASKWAIT
                block_ctx.have_task_id_resumable = false;
#endif
            }
            __syncthreads();
        }

        if (have_execute_task) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            if (threadIdx.x == 0) {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &block_ctx.cached_task_header;
                dst_hdr->generation = load_L2_u16t(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2_u16t(&src_hdr->parent_generation);
                dst_hdr->total_child_count = load_L2(&src_hdr->total_child_count);
            }
            __syncthreads();
#endif
            
#ifdef PROFILE
            if (threadIdx.x == 0) {
                if (working_time_idx < MAX_PROFILE_DATA) {
                    working_time[blockIdx.x][working_time_idx] = get_global_time();
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
            // if(threadIdx.x == 0) printf("finish_execute_task: %d\n", tid);
        }
        __syncthreads();
#ifdef PROFILE
        if (threadIdx.x == 0) {
            if (working_time_idx < MAX_PROFILE_DATA) {
                working_time[blockIdx.x][working_time_idx] = get_global_time();
                working_time_idx++;
            }
        }
#endif

        int total_count =
#ifdef GTAP_ASSUME_NO_TASKWAIT
            block_ctx.task_id_generated_count;
#else
            (block_ctx.have_task_id_resumable ? 1 : 0) + block_ctx.task_id_generated_count;
#endif
        int push_total = max(total_count - 1, 0);
        push(&block_ctx, push_total, &execute_task_id);
        if (threadIdx.x == 0) {
            have_execute_task = (total_count > 0);
        }
    }
#ifdef DEBUG
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
