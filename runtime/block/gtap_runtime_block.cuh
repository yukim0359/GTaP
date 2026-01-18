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
    // Info of current task
    uint16_t   generation;
    uint16_t   state;
    // Info of parent task
    int   parent_tid;
    uint16_t   parent_generation;
    // Info of child tasks
#ifdef GTAP_ASSUME_NO_TASKWAIT
    int   child_ids[0];
#else
    int   child_ids[GTAP_MAX_CHILD_TASKS];
    int   total_child_count;
    int   waiting_child_count;
#endif
};

// Grouped context passed to task functions to reduce parameter clutter
struct TaskContext {
    int task_id_generated_count;
    bool have_task_id_resumable;
    int id_list_alloc_pos;  // position to allocate next ID from
    int id_list_free_pos_stale;  // stale copy of free_pos to avoid L2 load on every allocation
    int task_id_resumable;
    TaskHeader cached_task_header;  // cached task header for reuse in task function
};

struct BlockTaskQueue {
    int queue[QUEUE_SIZE];
    int queue_head;
    // int queue_tail;
    int count;
    int queue_lock;
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
__device__ int* d_task_id_generated;  // [GTAP_GRID_SIZE * GTAP_MAX_CHILD_TASKS]
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

__device__ __forceinline__ int get_task_id_from_block_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int idx = old_alloc % GTAP_MAX_TASKS_PER_BLOCK;
    int block_id = (tid_list - d_task_id_lists);  // Compute once, used in both branches
    int id;
    if (old_alloc < GTAP_MAX_TASKS_PER_BLOCK) {
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
    return id;
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
    
    int* d_task_id_generated_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_ptr), task_id_array_size));
    CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    CUDA_TRY(cudaMemcpyToSymbol(d_block_task_queues, &d_block_task_queues_ptr, sizeof(BlockTaskQueue*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated, &d_task_id_generated_ptr, sizeof(int*)));
    
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
    
    int* d_task_id_generated_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
    
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
    
    if (d_task_id_generated_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_id_generated_ptr));
    }
    
    return cudaGetLastError();
}

// For backward compatibility
template <typename TaskType>
cudaError_t init_task_runtime() {
    return __gtap_init_task_runtime();
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
    
    int* d_task_id_generated_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_ptr, d_task_id_generated, sizeof(int*)));
    
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
    if (d_task_id_generated_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_MAX_CHILD_TASKS;
        CUDA_TRY(cudaMemsetAsync(d_task_id_generated_ptr, 0, task_id_array_size, streams[4]));
    }
    
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

// For backward compatibility
template <typename TaskType>
cudaError_t reset_task_runtime() {
    return __gtap_reset_task_runtime();
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

// define pop, steal, push
__device__ __forceinline__ int pop(int* taskId, int* shared_tail) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    bool pop_success = false;
    while (true) {
        int old_queue_count = load_L2(&myQueue->count);
        if (old_queue_count <= 0) break;
        if (atomicCAS(&myQueue->count, old_queue_count, old_queue_count - 1) == old_queue_count) {
            pop_success = true;
            (*shared_tail)--;
            break;
        }
    }

    if (pop_success) {
        int pop_task_id = load_L2(&myQueue->queue[*shared_tail % QUEUE_SIZE]);
        *taskId = pop_task_id;
#ifdef DEBUG
        printf("pop: %d (block: %d)\n", pop_task_id, blockIdx.x);
#endif
    } else {
        *taskId = -1;
    }
    return pop_success;
}

template<TerminationMode M>
__device__ __forceinline__ int steal(int* taskId, bool prev_get_task) {
    int targetBlock;
    int old_head;
    bool steal_success = false;
    BlockTaskQueue* targetBq = nullptr;
    while (true) {
        targetBlock = get_random_blocknum(blockIdx.x);
        targetBq = &d_block_task_queues[targetBlock];
        if (atomicCAS(&targetBq->queue_lock, 0, 1) == 0) break;
    }
    while (true) {
        int old_queue_count = load_L2(&targetBq->count);
        if (old_queue_count <= 0) break;
        if (atomicCAS(&targetBq->count, old_queue_count, old_queue_count - 1) == old_queue_count) {
            steal_success = true;
            old_head = load_L2(&targetBq->queue_head);
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (!prev_get_task) atomicAdd(&d_active_worker_count, 1);
            }
            break;
        }
    }
    
    if (!steal_success) {
        unlock(&targetBq->queue_lock);
        *taskId = -1;
        return false;
    }
    
    int steal_task_id = load_L2(&targetBq->queue[old_head % QUEUE_SIZE]);
    *taskId = steal_task_id;
    
    targetBq->queue_head = old_head + 1;
    __threadfence();
    unlock(&targetBq->queue_lock);
#ifdef DEBUG
    printf("steal: %d (block: %d -> %d)\n", steal_task_id, targetBlock, blockIdx.x);
#endif
    return true;
}

// NOTE: the template parameter is not used
template<TerminationMode M>
__device__ __forceinline__ void push(
    TaskContext* ctx,
    int push_total,
    int* execute_task_id,
    int* shared_tail
) {
    BlockTaskQueue* myQueue = &d_block_task_queues[blockIdx.x];
    
    int old_tail = *shared_tail;
    if (threadIdx.x == 0) {
        int old_head = load_L2(&myQueue->queue_head);
        if (old_tail + push_total - old_head > QUEUE_SIZE - QUEUE_MARGIN) {
            atomicExch(&d_runtime_error_code, GTAP_ERROR_QUEUE_OVERFLOW);
            __trap();
        }
    }

    if (ctx->have_task_id_resumable) {
        *execute_task_id = ctx->task_id_resumable;
        for (int i = threadIdx.x; i < ctx->task_id_generated_count; i += blockDim.x) {
            int queue_idx = (old_tail + i) % QUEUE_SIZE;
            myQueue->queue[queue_idx] = get_task_id_generated(blockIdx.x, i);
#ifdef DEBUG
            printf("push: %d (block: %d)\n", get_task_id_generated(blockIdx.x, i), blockIdx.x);
#endif
        }
    } else {
        *execute_task_id = get_task_id_generated(blockIdx.x, 0);
        for (int i = threadIdx.x + 1; i < ctx->task_id_generated_count; i += blockDim.x) {
            int queue_idx = (old_tail + i - 1) % QUEUE_SIZE;
            myQueue->queue[queue_idx] = get_task_id_generated(blockIdx.x, i);
        }
    }
    // if (threadIdx.x < push_total) __threadfence();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        *shared_tail = old_tail + push_total;
        atomicAdd(&myQueue->count, push_total);
    }
}


__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state) {
    if (threadIdx.x == 0) {
        TaskHeader* hdr = &d_task_headers[tid];
        hdr->state = next_state;
#ifndef GTAP_ASSUME_NO_TASKWAIT
        hdr->total_child_count = load_L2(&hdr->total_child_count) + child_count;
        // NOTE: total_child_count can be updated during the execution of task function, so we cannot use cached value here
        hdr->waiting_child_count = child_count;
#endif
    }
}


extern "C" {
__device__ __forceinline__ int __gtap_get_task_state(int tid) {
    return load_L2_u16t(&d_task_headers[tid].state);
}

__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state, int unused_value) {
    __gtap_set_state_for_join(tid, child_count, next_state);
}

__device__ __forceinline__ int __gtap_get_child_task_id(int parent_tid, int child_index) {
    return load_L2(&d_task_headers[parent_tid].child_ids[child_index]);
}
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
    if (threadIdx.x == 0) {
        TaskHeader* cached_hdr = &ctx->cached_task_header;
        int parent_tid = cached_hdr->parent_tid;
        d_task_headers[tid].generation = cached_hdr->generation + 1;
        
        if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
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

/*extern "C"*/ __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*)
) {
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    int new_tid = get_task_id_from_block_pool(tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->total_child_count = 0;
    new_hdr->waiting_child_count = 0;
#endif
    
    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    d_task_headers[self_tid].child_ids[cached_hdr->total_child_count + *child_count] = new_tid;
    (*child_count)++;
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

/*extern "C"*/ __device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size
) {
    TaskIdList* tid_list = &d_task_id_lists[blockIdx.x];
    int new_tid = get_task_id_from_block_pool(tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->cached_task_header;
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->total_child_count = 0;
    new_hdr->waiting_child_count = 0;
#endif
    
    void* dest_task = __gtap_get_task_data(new_tid);
    memcpy(dest_task, task_data_ptr, task_data_size);
    // __gtap_copy_bytes(dest_task, task_data_ptr, task_data_size);

    int gen_idx = atomicAdd(&ctx->task_id_generated_count, 1);
    set_task_id_generated(blockIdx.x, gen_idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    d_task_headers[self_tid].child_ids[cached_hdr->total_child_count + *child_count] = new_tid;
    (*child_count)++;
#endif
}

extern "C" __device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size,
    int unused_value
) { __gtap_spawn_task_raw(ctx, self_tid, child_count, func, task_data_ptr, task_data_size); }

/*extern "C"*/ __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*)
) { 
    TaskHeader* initial_hdr = &d_task_headers[0];
    initial_hdr->func = func;
    initial_hdr->state = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->total_child_count = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)
    
    BlockTaskQueue* bq = &d_block_task_queues[blockIdx.x];
    bq->queue[0] = 0;
    __threadfence();
    // if (blockIdx.x == 0) atomicExch(&d_active_worker_count, 1);
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
    __shared__ int tail;
    __shared__ TaskContext block_ctx;
#ifdef PROFILE
    __shared__ int having_task_time_idx;
    __shared__ int working_time_idx;
#endif

    if (threadIdx.x == 0) {
        should_continue = true;
        have_execute_task = false;
        block_ctx.have_task_id_resumable = false;
        block_ctx.task_id_generated_count = 0;
        block_ctx.id_list_free_pos_stale = GTAP_MAX_TASKS_PER_BLOCK;
#ifdef PROFILE
        working_time_idx = 0;
#endif
        if (blockIdx.x == 0) {
            tail = 1;
            block_ctx.id_list_alloc_pos = 1;
            prev_get_task = true;
            BlockTaskQueue* q = &d_block_task_queues[blockIdx.x];
            store_L2(&q->count, 1);
#ifdef PROFILE
            having_task_time_idx = 1;
            having_task_time[blockIdx.x][0] = get_global_time();
#endif
        } else {
            tail = 0;
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
                    have_execute_task = pop(&execute_task_id, &tail);
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
                            if (d_block_task_queues[blockIdx.x].queue_head < tail) {
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
#ifndef GTAP_ASSUME_NO_TASKWAIT
                dst_hdr->total_child_count = load_L2(&src_hdr->total_child_count);
#endif
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
            void* task_data = __gtap_get_task_data(execute_task_id);
            // Read function pointer atomically (64-bit) via L2 cache
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &block_ctx);
            // __threadfence();
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

        int total_count = (block_ctx.have_task_id_resumable ? 1 : 0) + block_ctx.task_id_generated_count;
        int push_total = max(total_count - 1, 0);
        push<M>(&block_ctx, push_total, &execute_task_id, &tail);
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
