#pragma once

#include <cuda_runtime.h>
#include "../common/gtap_runtime_common.cuh"

#ifndef __GTAP_WORKER_IS_THREAD
#define __GTAP_WORKER_IS_THREAD
#endif

#ifndef GTAP_NUM_QUEUES
#define GTAP_NUM_QUEUES 1
#endif

#ifndef GTAP_MAX_TASKS_PER_WARP
#define GTAP_MAX_TASKS_PER_WARP 20000
#endif

#define QUEUE_SIZE (GTAP_MAX_TASKS_PER_WARP / GTAP_NUM_QUEUES)

#define GTAP_TOTAL_TASK_IDS_PER_WARP (QUEUE_SIZE * GTAP_NUM_QUEUES)

#define MAX_TASKS_GLOBAL (QUEUE_SIZE * GTAP_NUM_QUEUES * GTAP_GRID_SIZE * GTAP_NUM_WARPS)

inline constexpr size_t __gtap_max_task_size = GTAP_MAX_TASK_DATA_SIZE;

// #define DEBUG
// #define INIT_PROFILE

struct TaskContext;

struct TaskHeader {
    void (*func)(void* task, int tid, TaskContext* __ctx);  // function pointer
    // Info of current task
    uint16_t   generation;
    uint16_t   state;
    uint16_t   queue_idx;
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
    int queue_idx;  // the queue index of the current task
    int task_id_generated_count_by_queue_idx[GTAP_NUM_QUEUES];  // how many tasks of each queue index were generated
    int id_list_alloc_pos;  // position to allocate next ID from
    int id_list_free_pos_stale;  // stale copy of free_pos to avoid L2 load on every allocation
    TaskHeader task_headers[WARP_SIZE];  // task headers for each thread
};

struct WarpTaskQueue {
    int queue[QUEUE_SIZE];
    int top;           // Chase-Lev top (steal from here)
    int bottom;        // Chase-Lev bottom (push/pop here)
};

struct TaskIdList {
    int id_list[GTAP_TOTAL_TASK_IDS_PER_WARP];
    int id_list_free_pos;
};

// Exposed device globals
// Queues remain 2D pointer planes by kind; headers/tasks are 1D of size GTAP_NUM_QUEUES*MAX_TASKS_GLOBAL
// Note: d_task_data_bytes is now char* (byte array) to support type-erased task data (static allocation)
__device__ WarpTaskQueue** d_warp_task_queues;
__device__ TaskHeader* d_task_headers;
__device__ char* d_task_data_bytes;  // Type-erased: byte array storing task data statically
__device__ TaskIdList* d_task_id_lists;
__device__ int* d_task_id_generated_by_queue_idx;  // [GTAP_GRID_SIZE * GTAP_NUM_WARPS * GTAP_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS+1) * WARP_SIZE]
__device__ int d_first_task_finished;
__device__ int d_all_tasks_finished_flag;
__device__ int d_active_worker_count;

#ifdef PROFILE
__device__ long long having_task_time[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
__device__ long long working_time[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
__device__ int tasks_processed_count[GTAP_GRID_SIZE * GTAP_NUM_WARPS][MAX_PROFILE_DATA];
#endif

// Pre-computed strides for d_task_id_generated_by_queue_idx access
constexpr int TASK_ID_GEN_QUEUE_STRIDE = (GTAP_MAX_CHILD_TASKS + 1) * WARP_SIZE;
constexpr int TASK_ID_GEN_WARP_STRIDE = GTAP_NUM_QUEUES * TASK_ID_GEN_QUEUE_STRIDE;

// Helper function to access d_task_id_generated_by_queue_idx
__device__ __forceinline__ int get_task_id_generated(int warp_id_global, int queue_idx, int idx) {
    int offset = warp_id_global * TASK_ID_GEN_WARP_STRIDE + queue_idx * TASK_ID_GEN_QUEUE_STRIDE + idx;
    return d_task_id_generated_by_queue_idx[offset];
}

__device__ __forceinline__ void set_task_id_generated(int warp_id_global, int queue_idx, int idx, int task_id) {
    int offset = warp_id_global * TASK_ID_GEN_WARP_STRIDE + queue_idx * TASK_ID_GEN_QUEUE_STRIDE + idx;
    d_task_id_generated_by_queue_idx[offset] = task_id;
}

// Initialize only the free_pos fields (id_list is lazily initialized on first access)
__global__ void init_warp_id_pools_metadata() {
    int warp_id_in_block = get_warp_id_in_block();
    int lane = get_lane_id();
    if (warp_id_in_block < GTAP_NUM_WARPS && lane == 0) {
        int qid = blockIdx.x * GTAP_NUM_WARPS + warp_id_in_block;
        TaskIdList* tid_list = &d_task_id_lists[qid];
        tid_list->id_list_free_pos = GTAP_TOTAL_TASK_IDS_PER_WARP;
    }
    __threadfence();
}

__device__ __forceinline__ int get_task_id_from_warp_pool(TaskIdList* tid_list, int* id_list_alloc_pos, int* id_list_free_pos_stale) {
    constexpr int total_elements = GTAP_TOTAL_TASK_IDS_PER_WARP;
    int old_alloc = atomicAdd(id_list_alloc_pos, 1);
    int idx = old_alloc % total_elements;
    int warp_id_global = (tid_list - d_task_id_lists);  // Compute once, used in both branches
    int id;
    if (old_alloc < total_elements) {
        // First pass: compute new ID directly (no L2 load needed)
        id = warp_id_global * total_elements + idx;
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

__device__ __forceinline__ void release_task_id_to_warp_pool(int id) {
    int warp_id_global = id / GTAP_TOTAL_TASK_IDS_PER_WARP;
    TaskIdList* tid_list = &d_task_id_lists[warp_id_global];
    int old_free = atomicAdd(&tid_list->id_list_free_pos, 1);
    store_L2(&tid_list->id_list[old_free % GTAP_TOTAL_TASK_IDS_PER_WARP], id);
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
    #ifdef INIT_PROFILE
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    
    cudaEventRecord(start);
    #endif

    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queues_ptrptr), sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(pointer array, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, elapsed);
    #endif
    
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
        #ifdef INIT_PROFILE
        cudaEventRecord(start);
        #endif
        WarpTaskQueue* plane_ptr = nullptr;
        CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&plane_ptr), sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
        cudaEventRecord(start);
        #endif
        CUDA_TRY(cudaMemset(plane_ptr, 0, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemset(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
        #endif
        h_warpTaskQueues_planes[k] = plane_ptr;
    }
    
    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemcpy(d_warp_task_queues_ptrptr, h_warpTaskQueues_planes, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyHostToDevice));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * MAX_TASKS_GLOBAL));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * MAX_TASKS_GLOBAL, elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * MAX_TASKS_GLOBAL));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * MAX_TASKS_GLOBAL, elapsed);
    cudaEventRecord(start);
    #endif

    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = GTAP_MAX_TASK_DATA_SIZE;
    size_t task_data_size = max_task_size * MAX_TASKS_GLOBAL;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);  
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemset(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif

    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * GTAP_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS + 1) * WARP_SIZE;
    CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_generated_by_queue_idx_ptr), task_id_array_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(task_id_generated, %zu bytes): %.3f ms\n", task_id_array_size, elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(task_id_generated, %zu bytes): %.3f ms\n", task_id_array_size, elapsed);
    cudaEventRecord(start);
    #endif

    CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queues, &d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue**)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_warp_task_queues): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_headers): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_data_bytes): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_lists): %.3f ms\n", elapsed); 
    cudaEventRecord(start);
    #endif
    CUDA_TRY(cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx, &d_task_id_generated_by_queue_idx_ptr, sizeof(int*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_generated_by_queue_idx): %.3f ms\n", elapsed);
    #endif

    free(h_warpTaskQueues_planes);

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    int zero = 0;
    CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
    #endif
    // Initialize d_active_worker_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_active_worker_count): %.3f ms\n", elapsed);
    #endif

#ifdef PROFILE
    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    CUDA_TRY(gtap_memset_symbol(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    CUDA_TRY(gtap_memset_symbol(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    CUDA_TRY(gtap_memset_symbol(tasks_processed_count, 0, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
#endif

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<GTAP_GRID_SIZE, GTAP_NUM_WARPS * WARP_SIZE>>>();
    CUDA_TRY(cudaDeviceSynchronize());
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  init_warp_id_pools_metadata kernel: %.3f ms\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("=== init_task_runtime profiling complete ===\n\n");
    #endif
    
    return cudaGetLastError();
}

cudaError_t __gtap_finalize_task_runtime() {
    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));
    
    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    if (d_warp_task_queues_ptrptr != nullptr) {
        CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyDeviceToHost));
        
        // Free each queue plane
        for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
            if (h_warpTaskQueues_planes[k] != nullptr) {
                CUDA_TRY(cudaFree(h_warpTaskQueues_planes[k]));
            }
        }
    }
    free(h_warpTaskQueues_planes);
    
    // Free queue pointer array
    if (d_warp_task_queues_ptrptr != nullptr) {
        CUDA_TRY(cudaFree(d_warp_task_queues_ptrptr));
    }
    
    // Free other allocated memory
    if (d_task_headers_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_headers_ptr));
    }
    
    if (d_task_data_bytes_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }
    
    if (d_task_id_lists_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }
    
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        CUDA_TRY(cudaFree(d_task_id_generated_by_queue_idx_ptr));
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
    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));
    
    TaskHeader* d_task_headers_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));
    
    char* d_task_data_bytes_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));
    
    TaskIdList* d_task_id_lists_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));
    
    int* d_task_id_generated_by_queue_idx_ptr = nullptr;
    CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_generated_by_queue_idx_ptr, d_task_id_generated_by_queue_idx, sizeof(int*)));
    
    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyDeviceToHost));
    
    // Clear task queues
    for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
        if (h_warpTaskQueues_planes[k] != nullptr) {
            CUDA_TRY(cudaMemset(h_warpTaskQueues_planes[k], 0, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
        }
    }
    free(h_warpTaskQueues_planes);
    
    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        CUDA_TRY(cudaMemset(d_task_headers_ptr, 0, sizeof(TaskHeader) * MAX_TASKS_GLOBAL));
    }
    
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t max_task_size = GTAP_MAX_TASK_DATA_SIZE;
        size_t task_data_size = max_task_size * MAX_TASKS_GLOBAL;
        CUDA_TRY(cudaMemset(d_task_data_bytes_ptr, 0, task_data_size));
    }
    
    // Reset task ID lists
    if (d_task_id_lists_ptr != nullptr) {
        CUDA_TRY(cudaMemset(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
    }
    
    // Clear task ID generated array
    if (d_task_id_generated_by_queue_idx_ptr != nullptr) {
        size_t task_id_array_size = sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * GTAP_NUM_QUEUES * (GTAP_MAX_CHILD_TASKS + 1) * WARP_SIZE;
        CUDA_TRY(cudaMemset(d_task_id_generated_by_queue_idx_ptr, 0, task_id_array_size));
    }
    
    // Reset global state
    int zero = 0;
    CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    int one = 1;
    CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));

    // Reset profile data if enabled
    #ifdef PROFILE
    CUDA_TRY(gtap_memset_symbol(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    CUDA_TRY(gtap_memset_symbol(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    CUDA_TRY(gtap_memset_symbol(tasks_processed_count, 0, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA));
    #endif
    
    // Reinitialize warp ID pools metadata
    init_warp_id_pools_metadata<<<GTAP_GRID_SIZE, GTAP_NUM_WARPS * WARP_SIZE>>>();
    CUDA_TRY(cudaDeviceSynchronize());
    
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
cudaError_t get_warp_having_task_time_data(long long* host_having_task_time) {
    return cudaMemcpyFromSymbol(host_having_task_time, having_task_time, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
}

cudaError_t get_warp_working_time_data(long long* host_working_time) {
    return cudaMemcpyFromSymbol(host_working_time, working_time, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
}

cudaError_t get_warp_tasks_processed_count_data(int* host_counts) {
    return cudaMemcpyFromSymbol(host_counts, tasks_processed_count, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
}

cudaError_t get_single_warp_having_task_time_data(int warp_global_id, long long* host_having_task_time, int max_samples) {
    long long* temp = (long long*)malloc(sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (!temp) return cudaErrorMemoryAllocation;
    cudaError_t st = cudaMemcpyFromSymbol(temp, having_task_time, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (st != cudaSuccess) { free(temp); return st; }
    for (int i = 0; i < max_samples && i < MAX_PROFILE_DATA; i++) host_having_task_time[i] = temp[warp_global_id * MAX_PROFILE_DATA + i];
    free(temp);
    return cudaSuccess;
}

cudaError_t get_single_warp_working_time_data(int warp_global_id, long long* host_working_time, int max_samples) {
    long long* temp = (long long*)malloc(sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (!temp) return cudaErrorMemoryAllocation;
    cudaError_t st = cudaMemcpyFromSymbol(temp, working_time, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (st != cudaSuccess) { free(temp); return st; }
    for (int i = 0; i < max_samples && i < MAX_PROFILE_DATA; i++) host_working_time[i] = temp[warp_global_id * MAX_PROFILE_DATA + i];
    free(temp);
    return cudaSuccess;
}

cudaError_t get_single_warp_tasks_processed_count_data(int warp_global_id, int* host_counts, int max_samples) {
    int* temp = (int*)malloc(sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (!temp) return cudaErrorMemoryAllocation;
    cudaError_t st = cudaMemcpyFromSymbol(temp, tasks_processed_count, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA);
    if (st != cudaSuccess) { free(temp); return st; }
    for (int i = 0; i < max_samples && i < MAX_PROFILE_DATA; i++) host_counts[i] = temp[warp_global_id * MAX_PROFILE_DATA + i];
    free(temp);
    return cudaSuccess;
}

__global__ void get_final_warp_having_task_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (having_task_time[wid][i] > 0) count++;
        }
        indices[wid] = count;
    }
}

__global__ void get_final_warp_working_time_indices(int* indices) {
    if (threadIdx.x == 0) {
        int wid = blockIdx.x;
        int count = 0;
        for (int i = 0; i < MAX_PROFILE_DATA; i++) {
            if (working_time[wid][i] > 0) count++;
        }
        indices[wid] = count;
    }
}
#endif

// Chase-Lev style sequential pop/steal operations

// Chase-Lev popBottom (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty)
__device__ __forceinline__ int pop_single_chase_lev(WarpTaskQueue* q) {
    int b = q->bottom - 1;
    store_L2(&q->bottom, b);
    __threadfence();
    int t = load_L2(&q->top);
    int size = b - t;
    
    if (size < 0) {
        q->bottom = t;
        return -1;
    }
    
    int task_id = load_L2(&q->queue[b % QUEUE_SIZE]);
    
    if (size > 0) {
        return task_id;
    }
    
    if (atomicCAS(&q->top, t, t + 1) != t) {
        // Lost race to stealer
        task_id = -1;
    }
    store_L2(&q->bottom, t + 1);
    return task_id;
}

// Sequential pop using chase-lev (repeats single pops)
__device__ __forceinline__ int pop_chase_lev(int* execute_task_id, int max_count_to_pop, int epaq_idx) {
    int lane = get_lane_id();
    WarpTaskQueue* myQueue = &d_warp_task_queues[epaq_idx][get_warp_id_global()];
    int pop_count = 0;
    
    for (int i = 0; i < max_count_to_pop; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = pop_single_chase_lev(myQueue);
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);
        
        if (task_id == -1) break;
        
        // Assign to lane (filling from high lanes: WARP_SIZE-max_count_to_pop, ...)
        int target_lane = WARP_SIZE - max_count_to_pop + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef DEBUG
            printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, epaq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        pop_count++;
    }
    
    return pop_count;
}

// Chase-Lev steal (single item) - called only by lane 0
// Returns task_id on success, -1 on failure (Empty or Abort)
__device__ __forceinline__ int steal_single_chase_lev(WarpTaskQueue* q) {
    int t = load_L2(&q->top);
    __threadfence();
    int b = load_L2(&q->bottom);
    
    int size = b - t;
    if (size <= 0) return -1;
    
    int task_id = load_L2(&q->queue[t % QUEUE_SIZE]);
    
    if (atomicCAS(&q->top, t, t + 1) != t) {
        return -1;  // Abort - lost race
    }
    
    return task_id;
}

// Sequential steal using chase-lev (repeats single steals)
template<TerminationMode M>
__device__ __forceinline__ int steal_chase_lev(int* execute_task_id, int max_count_to_steal, int epaq_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    WarpTaskQueue* targetWq = nullptr;
    int steal_count = 0;
    bool active_count_incremented = false;
    
    // Select a random victim (lane 0 only)
    if (lane == 0) {
        target_warp_id_global = get_random_warpnum_global(warp_id_global);
        targetWq = &d_warp_task_queues[epaq_idx][target_warp_id_global];
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    targetWq = &d_warp_task_queues[epaq_idx][target_warp_id_global];
    
    // Sequential steals using chase-lev
    for (int i = 0; i < max_count_to_steal; i++) {
        int task_id = -1;
        if (lane == 0) {
            task_id = steal_single_chase_lev(targetWq);
        }
        task_id = __shfl_sync(0xFFFFFFFFu, task_id, 0);
        
        if (task_id == -1) break;
        
        // Increment active worker count on first successful steal
        if (M == TERMINATE_ON_ALL_TASKS_FINISH && !active_count_incremented && !prev_get_task) {
            if (lane == 0) atomicAdd(&d_active_worker_count, 1);
            active_count_incremented = true;
        }
        
        // Assign to lane (filling from high lanes: WARP_SIZE-max_count_to_steal, ...)
        int target_lane = WARP_SIZE - max_count_to_steal + i;
        if (lane == target_lane) {
            *execute_task_id = task_id;
#ifdef DEBUG
            printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", task_id, epaq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        steal_count++;
    }
    
    return steal_count;
}

// Chase-Lev pushBottom (multiple items)
// NOTE: the template parameter is not used
template<TerminationMode M>
__device__ __forceinline__ void push_batch (
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count
) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int k_max = 0;
    int max_gen = -1;
    int all_generated_count = 0;
    if (lane == 0) {
        for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
            int cnt = ctx->task_id_generated_count_by_queue_idx[k];
            all_generated_count += cnt;
            if (cnt > max_gen) {
                max_gen = cnt;
                k_max = k;
            }
        }
        ctx->queue_idx = k_max;
    }
    all_generated_count = __shfl_sync(0xFFFFFFFFu, all_generated_count, 0);
    if (all_generated_count == 0) {
        *execute_task_count = 0;
        return;
    }
    k_max = __shfl_sync(0xFFFFFFFFu, k_max, 0);
    max_gen = __shfl_sync(0xFFFFFFFFu, max_gen, 0);

    *execute_task_count = max(0, min(WARP_SIZE, max_gen));
    if (lane < *execute_task_count) {
        *execute_task_id = get_task_id_generated(warp_id_global, k_max, lane);
#ifdef DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }

    #pragma unroll
    for (int kind = 0; kind < GTAP_NUM_QUEUES; ++kind) {
        int first_idx_to_push = (kind == k_max) ? *execute_task_count : 0;
        int push_cnt = ctx->task_id_generated_count_by_queue_idx[kind] - first_idx_to_push;
        if (push_cnt <= 0) continue;

        WarpTaskQueue* q = &d_warp_task_queues[kind][warp_id_global];
        
        int b = load_L2(&q->bottom);
        int t = load_L2(&q->top);
        
        if (lane == 0) {
            int size = b - t;
            if (size + push_cnt >= QUEUE_SIZE - QUEUE_MARGIN) {
                atomicExch(&d_runtime_error_code, GTAP_ERROR_QUEUE_OVERFLOW);
                __trap();
            }
        }

        for (int j = lane; j < push_cnt; j += WARP_SIZE) {
            int idx_to_push = (b + j) % QUEUE_SIZE;
            int val = get_task_id_generated(warp_id_global, kind, first_idx_to_push + j);
            q->queue[idx_to_push] = val;
#ifdef DEBUG
            printf("push_task_id: %d to %d (kind %d) in lane %d of warp %d of block %d\n", val, idx_to_push, kind, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        }
        __threadfence();
        __syncwarp();
        
        if (lane == 0) {
            store_L2(&q->bottom, b + push_cnt);
        }
    }
}

extern "C" {
// Get the current state of a task (reads from TaskHeader)
__device__ __forceinline__ int __gtap_get_task_state(int tid) {
    return load_L2_u16t(&d_task_headers[tid].state);
}

__device__ __forceinline__ void __gtap_set_state_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join >= GTAP_NUM_QUEUES) {
        atomicExch(&d_runtime_error_code, GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN);
        __trap();
    }
    TaskHeader* hdr = &d_task_headers[tid];
    hdr->state = next_state;
    hdr->queue_idx = queue_idx_after_join;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    hdr->total_child_count = load_L2(&hdr->total_child_count) + child_count;
    // NOTE: total_child_count can be updated during the execution of task function, so we cannot use cached value here
    hdr->waiting_child_count = child_count;
#endif
}

// Get the child task ID by index (for result retrieval after taskwait)
__device__ __forceinline__ int __gtap_get_child_task_id(int parent_tid, int child_index) {
    return load_L2(&d_task_headers[parent_tid].child_ids[child_index]);
}
}

#ifndef GTAP_ASSUME_NO_TASKWAIT
__device__ __forceinline__ int notify_parent(int parentId, TaskContext* ctx) {
    TaskHeader* parent_hdr = &d_task_headers[parentId];
    int rem = atomicSub(&parent_hdr->waiting_child_count, 1);
#ifdef DEBUG
    int lane = get_lane_id();
    printf("notify_parent: %d (remaining child count: %d) in lane %d of warp %d of block %d\n", parentId, rem, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    if (rem == 1) {
        int parent_queue_idx = load_L2_u16t(&parent_hdr->queue_idx);
        int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[parent_queue_idx], 1);
        set_task_id_generated(get_warp_id_global(), parent_queue_idx, idx, parentId);
    }
    return rem;
}
#endif

extern "C" __device__ __forceinline__ void __gtap_finish_task(int tid, TaskContext* ctx) {
#ifdef DEBUG
    printf("finish_task: %d in lane %d of warp %d of block %d\n", tid, get_lane_id(), get_warp_id_in_block(), blockIdx.x);
#endif
    int lane = get_lane_id();
    TaskHeader* cached_hdr = &ctx->task_headers[lane];
    int parent_tid = cached_hdr->parent_tid;
    d_task_headers[tid].generation = cached_hdr->generation + 1;
    
    if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
#ifndef GTAP_ASSUME_NO_TASKWAIT
        notify_parent(parent_tid, ctx);
        // If child tasks are joined, release the task IDs of child tasks
        int* child_ids = d_task_headers[tid].child_ids;
        int child_count = cached_hdr->total_child_count;
        for (int i = 0; i < child_count; i++) {
            release_task_id_to_warp_pool(child_ids[i]);
        }
        // If the task is not waited by the parent, release the task ID of the task itself
        // TODO: Is this really necessary?
        // if (parent_waiting_child_count_old <= 0) {
        //     release_task_id_to_warp_pool(tid);
        // }
#else
        // NO_TASKWAIT: no need to notify parent or release child IDs
        release_task_id_to_warp_pool(tid);
#endif
    } else {
        release_task_id_to_warp_pool(tid);
    }
    
    if (tid == 0) {
        store_L2(&d_first_task_finished, 1);
#ifdef DEBUG
        int lane = get_lane_id();
        printf("first task finished in lane %d of warp %d of block %d\n", lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
}

// Allocates task ID, sets up TaskHeader, returns task data pointer
// Caller stores task data fields after this call
extern "C" __device__ __forceinline__ void* __gtap_spawn_task(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    int child_queue_idx
) {
    if (child_queue_idx >= GTAP_NUM_QUEUES) {
        atomicExch(&d_runtime_error_code, GTAP_ERROR_INVALID_QUEUE_IDX);
        __trap();
    }
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int new_tid = get_task_id_from_warp_pool(&d_task_id_lists[warp_id_global], &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);

    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->task_headers[lane];
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->queue_idx = child_queue_idx;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->total_child_count = 0;
    new_hdr->waiting_child_count = 0;
#endif
    
    int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[child_queue_idx], 1);
    set_task_id_generated(warp_id_global, child_queue_idx, idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    d_task_headers[self_tid].child_ids[cached_hdr->total_child_count + *child_count] = new_tid;
    (*child_count)++;
#endif
    return __gtap_get_task_data(new_tid);
}


// Non-template version that takes a pointer and size for compiler-generated code
extern "C" __device__ __forceinline__ void __gtap_spawn_task_raw(
    TaskContext* ctx,
    int self_tid,
    int* child_count,
    void (*func)(void*, int, TaskContext*),
    const void* task_data_ptr,
    size_t task_data_size,
    int child_queue_idx
) {
    if (child_queue_idx >= GTAP_NUM_QUEUES) {
        atomicExch(&d_runtime_error_code, GTAP_ERROR_INVALID_QUEUE_IDX);
        __trap();
    }

    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    TaskIdList* tid_list = &d_task_id_lists[warp_id_global];
    int new_tid = get_task_id_from_warp_pool(tid_list, &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    
    TaskHeader* new_hdr = &d_task_headers[new_tid];
    TaskHeader* cached_hdr = &ctx->task_headers[lane];
    new_hdr->func = func;
    new_hdr->state = 0;
    new_hdr->queue_idx = child_queue_idx;
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    new_hdr->total_child_count = 0;
    new_hdr->waiting_child_count = 0;
#endif

    // Copy task data atomically word-by-word
    void* dest_task = __gtap_get_task_data(new_tid);
    memcpy(dest_task, task_data_ptr, task_data_size);
    // __gtap_copy_bytes(dest_task, task_data_ptr, task_data_size);
    
    int idx = atomicAdd(&ctx->task_id_generated_count_by_queue_idx[child_queue_idx], 1);
    set_task_id_generated(warp_id_global, child_queue_idx, idx, new_tid);
#ifndef GTAP_ASSUME_NO_TASKWAIT
    d_task_headers[self_tid].child_ids[cached_hdr->total_child_count + *child_count] = new_tid;
    (*child_count)++;
#endif
}

// Non-template version for compiler-generated code using raw pointer and size
// __gtap_push_initial_task: Device function to push initial task
// This function is called from compiler-generated kernel code
extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int new_tid = 0;

    TaskHeader* initial_hdr = &d_task_headers[new_tid];
    initial_hdr->func = func;
    initial_hdr->state = 0;
    initial_hdr->queue_idx = initial_queue_idx;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->total_child_count = 0;
    initial_hdr->waiting_child_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)

    WarpTaskQueue* wq = &d_warp_task_queues[initial_queue_idx][warp_id_global];
    wq->queue[0] = new_tid;
    __threadfence();
    // atomicExch(&d_active_worker_count, 1);
}

template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    int warp_id_in_block = get_warp_id_in_block();
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);
    bool should_continue = true;

    __shared__ TaskContext warp_contexts[GTAP_NUM_WARPS];

#ifdef PROFILE
    __shared__ int having_time_idx[GTAP_NUM_WARPS];
    __shared__ int working_time_idx[GTAP_NUM_WARPS];
    if (lane == 0) {
        if (warp_id_global == 0) having_time_idx[warp_id_in_block] = 1;
        else having_time_idx[warp_id_in_block] = 0;
        working_time_idx[warp_id_in_block] = 0;
    }
#endif

    if (lane == 0) {
        warp_contexts[warp_id_in_block].queue_idx = 0;
        warp_contexts[warp_id_in_block].id_list_free_pos_stale = GTAP_TOTAL_TASK_IDS_PER_WARP;
        #pragma unroll
        for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
            warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
        }
        if (warp_id_global == 0) {
#ifdef PROFILE
            having_task_time[warp_id_global][0] = get_global_time();
#endif
            warp_contexts[0].id_list_alloc_pos = 1;
            // Chase-Lev: set bottom = 1 (initial task at position 0)
            WarpTaskQueue* q = &d_warp_task_queues[0][0];
            q->bottom = 1;
        } else {
            warp_contexts[warp_id_in_block].id_list_alloc_pos = 0;
        }
    }
    __syncwarp();
    
    while (should_continue) {
        if (execute_task_count == 0) {
            #pragma unroll
            for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
                if (prev_get_task && execute_task_count < WARP_SIZE) {
                    int remaining = WARP_SIZE - execute_task_count;
                    int pop_count = pop_chase_lev(&execute_task_id, remaining, warp_contexts[warp_id_in_block].queue_idx);
                    execute_task_count += pop_count;
                }
                if (execute_task_count < WARP_SIZE) {
                    int remaining = WARP_SIZE - execute_task_count;
                    int steal_count = steal_chase_lev<M>(&execute_task_id, remaining, warp_contexts[warp_id_in_block].queue_idx, prev_get_task);
                    execute_task_count += steal_count;
                }
                if (execute_task_count != 0) break;
                warp_contexts[warp_id_in_block].queue_idx = (warp_contexts[warp_id_in_block].queue_idx + 1) % GTAP_NUM_QUEUES;
            }
        }

        if (execute_task_count == 0) {
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) {
                    if (prev_get_task) {
                        int active_worker_count = atomicSub(&d_active_worker_count, 1) - 1;
                        if (active_worker_count == 0) {
                            bool all_tasks_finished = 1;
                            #pragma unroll
                            for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
                                // Chase-Lev: check if queue is empty (top >= bottom)
                                WarpTaskQueue* q = &d_warp_task_queues[k][warp_id_global];
                                if (q->top < q->bottom) {
                                    all_tasks_finished = 0;
                                    break;
                                }
                            }
                            atomicExch(&d_all_tasks_finished_flag, all_tasks_finished);
                        }
                    }
                }
                __syncwarp();
            }
#ifdef PROFILE
            if (lane == 0) {
                if (prev_get_task && having_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    having_task_time[warp_id_global][having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = false;
            if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                if (lane == 0) should_continue = (load_L2(&d_all_tasks_finished_flag) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            } else {
                if (lane == 0) should_continue = (load_L2(&d_first_task_finished) == 0);
                should_continue = __shfl_sync(0xFFFFFFFFu, should_continue, 0);
            }
            continue;
        } else {
#ifdef PROFILE
            if (lane == 0) {
                if (!prev_get_task && having_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    having_task_time[warp_id_global][having_time_idx[warp_id_in_block]] = get_global_time();
                    having_time_idx[warp_id_in_block]++;
                }
            }
            __syncwarp();
#endif
            prev_get_task = true;
            if (lane == 0) {
                for (int k = 0; k < GTAP_NUM_QUEUES; ++k) warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            // unsigned active_mask = __activemask();
            // Copy task header to TaskContext for reuse in task function (using L2 load)
            {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &warp_contexts[warp_id_in_block].task_headers[lane];
                dst_hdr->generation = load_L2_u16t(&src_hdr->generation);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2_u16t(&src_hdr->parent_generation);
#ifndef GTAP_ASSUME_NO_TASKWAIT
                dst_hdr->total_child_count = load_L2(&src_hdr->total_child_count);
#endif
            }
            // __syncwarp(active_mask);
            
#ifdef PROFILE
            if (lane == 0) {
                if (working_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    working_time[warp_id_global][working_time_idx[warp_id_in_block]] = get_global_time();
                    tasks_processed_count[warp_id_global][working_time_idx[warp_id_in_block]] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                }
            }
#endif
            // Use non-template version to avoid TaskType dependency
            void* task_data = __gtap_get_task_data(execute_task_id);
            // printf("task_data: %p\n", task_data);
            // if (lane == 0) {
            //     printf("execute_task_loop: execute_task_id = %d, d_task_headers[%d].func = %p\n", execute_task_id, execute_task_id, d_task_headers[execute_task_id].func);
            // }
            // Read function pointer atomically (64-bit)
            void* func_ptr = load_L2_ptr(reinterpret_cast<void**>(&d_task_headers[execute_task_id].func));
            void (*task_func)(void*, int, TaskContext*) = reinterpret_cast<void (*)(void*, int, TaskContext*)>(func_ptr);
            task_func(task_data, execute_task_id, &warp_contexts[warp_id_in_block]);
#ifdef DEBUG
            printf("executed_task_id: %d in lane %d of warp %d of block %d\n", execute_task_id, lane, warp_id_in_block, blockIdx.x);
#endif
            __threadfence();
        }
        __syncwarp();
#ifdef PROFILE
        if (lane == 0) {
            if (working_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                working_time[warp_id_global][working_time_idx[warp_id_in_block]] = get_global_time();
                tasks_processed_count[warp_id_global][working_time_idx[warp_id_in_block]] = execute_task_count;
                working_time_idx[warp_id_in_block]++;
            }
        }
        __syncwarp();
#endif

        push_batch<M>(
            &warp_contexts[warp_id_in_block], &execute_task_id,
            &execute_task_count
        );
    }
#ifdef DEBUG
    if (lane == 0) printf("execute_task_loop: end (warp_id_global = %d)\n", warp_id_global);
#endif
}

// Non-template device-side wrapper
extern "C" __device__ __forceinline__ void __gtap_execute_task_loop_device() {
#ifdef GTAP_TERMINATE_ON_FIRST_TASK_FINISH
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_FIRST_TASK_FINISH>();
#else
    __gtap_execute_task_loop_device_impl<TerminationMode::TERMINATE_ON_ALL_TASKS_FINISH>();
#endif
}
