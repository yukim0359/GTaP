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

#define GTAP_QUEUE_SIZE (GTAP_MAX_TASKS_PER_WARP / GTAP_NUM_QUEUES)

#define GTAP_TOTAL_TASK_IDS_PER_WARP (GTAP_QUEUE_SIZE * GTAP_NUM_QUEUES)

#define GTAP_MAX_TASKS_GLOBAL (GTAP_QUEUE_SIZE * GTAP_NUM_QUEUES * GTAP_GRID_SIZE * GTAP_NUM_WARPS)

#include "gtap_thread_core.cuh"

struct WarpTaskQueue {
    int count;
    int queue_lock;
    int queue_head;
    int queue_head_stale;
    int queue[GTAP_QUEUE_SIZE];
};

__constant__ WarpTaskQueue** d_warp_task_queues;

__device__ __forceinline__ void reserve_unpublished_task_id(TaskContext* ctx, int queue_idx, int task_id) {
    WarpTaskQueue* q = &d_warp_task_queues[queue_idx][get_warp_id_global()];
    int old_tail = atomicAdd(&ctx->tail_by_queue_idx[queue_idx], 1);
    int head = load_L2(&q->queue_head);
    if (old_tail + 1 - head > GTAP_QUEUE_SIZE - GTAP_QUEUE_MARGIN) {
        gtap_record_runtime_error_and_trap(
            GTAP_ERROR_QUEUE_OVERFLOW, get_warp_id_global(), task_id, queue_idx,
            old_tail + 1 - head, GTAP_QUEUE_SIZE - GTAP_QUEUE_MARGIN, __LINE__);
    }
    q->queue[old_tail % GTAP_QUEUE_SIZE] = task_id;
    atomicAdd(&ctx->task_id_generated_count_by_queue_idx[queue_idx], 1);
}

cudaError_t __gtap_init_task_runtime() {
    GTAP_CUDA_TRY(gtap_init_runtime_error_report());

    constexpr int NUM_STREAMS = GTAP_NUM_QUEUES + 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    #ifdef INIT_PROFILE
    printf("\n=== init_task_runtime detailed profiling ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    cudaEventRecord(start);
    #endif

    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_warp_task_queues_ptrptr), sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));

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
        GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&plane_ptr), sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMalloc(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
        cudaEventRecord(start);
        #endif
        GTAP_CUDA_TRY(cudaMemsetAsync(plane_ptr, 0, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, streams[k]));
        #ifdef INIT_PROFILE
        cudaEventRecord(stop, streams[k]);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("  cudaMemsetAsync(queue plane %d, %zu bytes): %.3f ms\n", k, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
        #endif
        h_warpTaskQueues_planes[k] = plane_ptr;
    }

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpy(d_warp_task_queues_ptrptr, h_warpTaskQueues_planes, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyHostToDevice));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpy(pointer array H->D, %zu bytes): %.3f ms\n", sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, elapsed);
    cudaEventRecord(start);
    #endif

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_headers_ptr), sizeof(TaskHeader) * GTAP_MAX_TASKS_GLOBAL));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_MAX_TASKS_GLOBAL, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_MAX_TASKS_GLOBAL, streams[GTAP_NUM_QUEUES]));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop, streams[GTAP_NUM_QUEUES]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(TaskHeaders, %zu bytes): %.3f ms\n", sizeof(TaskHeader) * GTAP_MAX_TASKS_GLOBAL, elapsed);
    cudaEventRecord(start);
    #endif

    char* d_task_data_bytes_ptr = nullptr;
    size_t max_task_size = gtap_host_task_data_stride();
    size_t task_data_size = max_task_size * GTAP_MAX_TASKS_GLOBAL;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_data_bytes_ptr), task_data_size));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[GTAP_NUM_QUEUES + 1]));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop, streams[GTAP_NUM_QUEUES + 1]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(Task data storage, %zu bytes): %.3f ms\n", task_data_size, elapsed);
    cudaEventRecord(start);
    #endif

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_task_id_lists_ptr), sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMalloc(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, streams[GTAP_NUM_QUEUES + 2]));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop, streams[GTAP_NUM_QUEUES + 2]);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemsetAsync(TaskIdLists, %zu bytes): %.3f ms\n", sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, elapsed);
    cudaEventRecord(start);
    #endif

    GTaPResultHandle* d_result_handles_ptr = nullptr;
    size_t result_handle_array_size = sizeof(GTaPResultHandle) * GTAP_RESULT_HANDLE_CAPACITY;
    GTAP_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_result_handles_ptr), result_handle_array_size));
    GTAP_CUDA_TRY(cudaMemsetAsync(d_result_handles_ptr, 0, result_handle_array_size, streams[GTAP_NUM_QUEUES + 3]));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_warp_task_queues, &d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue**)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_warp_task_queues): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_headers, &d_task_headers_ptr, sizeof(TaskHeader*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_headers): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_data_bytes, &d_task_data_bytes_ptr, sizeof(char*)));
    GTAP_CUDA_TRY(gtap_init_device_task_data_stride());
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_data_bytes): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_task_id_lists, &d_task_id_lists_ptr, sizeof(TaskIdList*)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_task_id_lists): %.3f ms\n", elapsed);
    cudaEventRecord(start);
    #endif
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_result_handles, &d_result_handles_ptr, sizeof(GTaPResultHandle*)));

    free(h_warpTaskQueues_planes);

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_result_handle_top, &zero, sizeof(int)));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemcpyToSymbol(d_first_task_finished): %.3f ms\n", elapsed);
    #endif
    // Initialize d_active_worker_count to 1 to prevent early termination
    // before the initial task is pushed by the master thread
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));
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
    GTAP_CUDA_TRY(gtap_memset_symbol_async(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[0]));
    GTAP_CUDA_TRY(gtap_memset_symbol_async(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(gtap_memset_symbol_async(tasks_processed_count, 0, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[2 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[0]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(cudaStreamSynchronize(streams[2 % NUM_STREAMS]));
    #ifdef INIT_PROFILE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  cudaMemset(profile data): %.3f ms\n", elapsed);
    #endif
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    #ifdef INIT_PROFILE
    cudaEventRecord(start);
    #endif
    init_warp_id_pools_metadata<<<GTAP_GRID_SIZE, GTAP_NUM_WARPS * GTAP_WARP_SIZE>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());
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
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));

    GTaPResultHandle* d_result_handles_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_result_handles_ptr, d_result_handles, sizeof(GTaPResultHandle*)));

    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyDeviceToHost));

        // Free each queue plane
        for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
            if (h_warpTaskQueues_planes[k] != nullptr) {
                GTAP_CUDA_TRY(cudaFree(h_warpTaskQueues_planes[k]));
            }
        }
    }
    free(h_warpTaskQueues_planes);

    if (d_warp_task_queues_ptrptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_warp_task_queues_ptrptr));
    }

    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_headers_ptr));
    }

    if (d_task_data_bytes_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_data_bytes_ptr));
    }

    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_task_id_lists_ptr));
    }

    if (d_result_handles_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaFree(d_result_handles_ptr));
    }

    GTAP_CUDA_TRY(gtap_finalize_runtime_error_report());

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
    gtap_reset_runtime_error_report_host();

    constexpr int NUM_STREAMS = GTAP_NUM_QUEUES + 4;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamCreate(&streams[i]));
    }

    // Get device pointers from symbols
    WarpTaskQueue** d_warp_task_queues_ptrptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_warp_task_queues_ptrptr, d_warp_task_queues, sizeof(WarpTaskQueue**)));

    TaskHeader* d_task_headers_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_headers_ptr, d_task_headers, sizeof(TaskHeader*)));

    char* d_task_data_bytes_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_data_bytes_ptr, d_task_data_bytes, sizeof(char*)));

    TaskIdList* d_task_id_lists_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_task_id_lists_ptr, d_task_id_lists, sizeof(TaskIdList*)));

    GTaPResultHandle* d_result_handles_ptr = nullptr;
    GTAP_CUDA_TRY(cudaMemcpyFromSymbol(&d_result_handles_ptr, d_result_handles, sizeof(GTaPResultHandle*)));

    // Get queue plane pointers from device
    WarpTaskQueue** h_warpTaskQueues_planes = reinterpret_cast<WarpTaskQueue**>(malloc(sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES));
    GTAP_CUDA_TRY(cudaMemcpy(h_warpTaskQueues_planes, d_warp_task_queues_ptrptr, sizeof(WarpTaskQueue*) * GTAP_NUM_QUEUES, cudaMemcpyDeviceToHost));

    // Clear task queues
    for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
        if (h_warpTaskQueues_planes[k] != nullptr) {
            GTAP_CUDA_TRY(cudaMemsetAsync(h_warpTaskQueues_planes[k], 0, sizeof(WarpTaskQueue) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, streams[k]));
        }
    }
    free(h_warpTaskQueues_planes);

    // Clear task headers
    if (d_task_headers_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_headers_ptr, 0, sizeof(TaskHeader) * GTAP_MAX_TASKS_GLOBAL, streams[GTAP_NUM_QUEUES]));
    }

    size_t max_task_size = gtap_host_task_data_stride();
    // Clear task data
    if (d_task_data_bytes_ptr != nullptr) {
        size_t task_data_size = max_task_size * GTAP_MAX_TASKS_GLOBAL;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_data_bytes_ptr, 0, task_data_size, streams[GTAP_NUM_QUEUES + 1]));
    }

    // Reset task ID lists
    if (d_task_id_lists_ptr != nullptr) {
        GTAP_CUDA_TRY(cudaMemsetAsync(d_task_id_lists_ptr, 0xFF, sizeof(TaskIdList) * GTAP_GRID_SIZE * GTAP_NUM_WARPS, streams[GTAP_NUM_QUEUES + 2]));
    }

    if (d_result_handles_ptr != nullptr) {
        size_t result_handle_array_size = sizeof(GTaPResultHandle) * GTAP_RESULT_HANDLE_CAPACITY;
        GTAP_CUDA_TRY(cudaMemsetAsync(d_result_handles_ptr, 0, result_handle_array_size, streams[GTAP_NUM_QUEUES + 3]));
    }

    // Reset global state
    int zero = 0;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_first_task_finished, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_all_tasks_finished_flag, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_runtime_error_code, &zero, sizeof(int)));
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_result_handle_top, &zero, sizeof(int)));
    int one = 1;
    GTAP_CUDA_TRY(cudaMemcpyToSymbol(d_active_worker_count, &one, sizeof(int)));

    // Reset profile data if enabled
    #ifdef PROFILE
    GTAP_CUDA_TRY(gtap_memset_symbol_async(having_task_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[0]));
    GTAP_CUDA_TRY(gtap_memset_symbol_async(working_time, 0, sizeof(long long) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[1 % NUM_STREAMS]));
    GTAP_CUDA_TRY(gtap_memset_symbol_async(tasks_processed_count, 0, sizeof(int) * GTAP_GRID_SIZE * GTAP_NUM_WARPS * MAX_PROFILE_DATA, streams[2 % NUM_STREAMS]));
    #endif

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamSynchronize(streams[i]));
    }

    // Reinitialize warp ID pools metadata
    init_warp_id_pools_metadata<<<GTAP_GRID_SIZE, GTAP_NUM_WARPS * GTAP_WARP_SIZE>>>();
    GTAP_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < NUM_STREAMS; ++i) {
        GTAP_CUDA_TRY(cudaStreamDestroy(streams[i]));
    }

    return cudaGetLastError();
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

// define pop_batch, steal_batch, push_batch
__device__ __forceinline__ int pop_batch(int* execute_task_id, int max_count_to_pop, int* tail, int epaq_idx) {
    int lane = get_lane_id();
    WarpTaskQueue* myQueue = &d_warp_task_queues[epaq_idx][get_warp_id_global()];
    int pop_count = 0;
    if (lane == 0) {
        while (true) {
            int old_queue_count = load_L2(&myQueue->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_pop, old_queue_count);
            if (atomicCAS(&myQueue->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                pop_count = claim;
                *tail -= claim;
                break;
            }
        }
    }
    pop_count = __shfl_sync(0xFFFFFFFFu, pop_count, 0);
    if (lane >= GTAP_WARP_SIZE - max_count_to_pop && lane < GTAP_WARP_SIZE - max_count_to_pop + pop_count) {
        int pop_task_id = load_L2(&myQueue->queue[(*tail + (lane - GTAP_WARP_SIZE + max_count_to_pop)) % GTAP_QUEUE_SIZE]);
#ifdef DEBUG
        printf("pop_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", pop_task_id, epaq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = pop_task_id;
    }
    return pop_count;
}

template<TerminationMode M>
__device__ __forceinline__ int steal_batch(int* execute_task_id, int max_count_to_steal, int epaq_idx, bool prev_get_task) {
    int warp_id_global = get_warp_id_global();
    int lane = get_lane_id();
    int target_warp_id_global = 0;
    int old_head = 0;
    int steal_count = 0;
    WarpTaskQueue* targetWq = nullptr;
    if (lane == 0) {
        unsigned lock_backoff_ns = 32;
        while (true) {
            target_warp_id_global = get_random_warpnum_global(warp_id_global);
            targetWq = &d_warp_task_queues[epaq_idx][target_warp_id_global];
            if (atomicCAS(&targetWq->queue_lock, 0, 1) == 0) break;
            __nanosleep(lock_backoff_ns);
            if (lock_backoff_ns < (1u << 12)) {
                lock_backoff_ns <<= 1u;
            }
        }
        while (true) {
            int old_queue_count = load_L2(&targetWq->count);
            if (old_queue_count <= 0) break;
            int claim = min(max_count_to_steal, old_queue_count);
            if (atomicCAS(&targetWq->count, old_queue_count, old_queue_count - claim) == old_queue_count) {
                if (M == TERMINATE_ON_ALL_TASKS_FINISH) {
                    if (!prev_get_task) atomicAdd(&d_active_worker_count, 1);
                }
                steal_count = claim;
                old_head = load_L2(&targetWq->queue_head);
                break;
            }
        }
    }
    steal_count = __shfl_sync(0xFFFFFFFFu, steal_count, 0);
    if (steal_count == 0) {
        if (lane == 0) atomicExch(&targetWq->queue_lock, 0);
        return 0;
    }
    target_warp_id_global = __shfl_sync(0xFFFFFFFFu, target_warp_id_global, 0);
    old_head = __shfl_sync(0xFFFFFFFFu, old_head, 0);
    if (lane >= GTAP_WARP_SIZE - max_count_to_steal && lane < GTAP_WARP_SIZE - max_count_to_steal + steal_count) {
        targetWq = &d_warp_task_queues[epaq_idx][target_warp_id_global];
        int steal_task_id = load_L2(&targetWq->queue[(old_head + (lane - GTAP_WARP_SIZE + max_count_to_steal)) % GTAP_QUEUE_SIZE]);
#ifdef DEBUG
        printf("steal_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", steal_task_id, epaq_idx, lane, get_warp_id_in_block(), blockIdx.x);
#endif
        *execute_task_id = steal_task_id;
    }
    __syncwarp();
    if (lane == 0) {
        targetWq->queue_head = old_head + steal_count;
        __threadfence();
        atomicExch(&targetWq->queue_lock, 0);
    }
    return steal_count;
}

__device__ __forceinline__ void push_batch (
    TaskContext* ctx,
    int* execute_task_id,
    int* execute_task_count,
    int* tail_by_queue_idx
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

    *execute_task_count = max(0, min(GTAP_WARP_SIZE, max_gen));
    WarpTaskQueue* direct_q = &d_warp_task_queues[k_max][warp_id_global];
    int direct_start = tail_by_queue_idx[k_max] - *execute_task_count;
    if (lane < *execute_task_count) {
        *execute_task_id = direct_q->queue[(direct_start + lane) % GTAP_QUEUE_SIZE];
#ifdef DEBUG
        printf("push_task_id: %d (kind %d) in lane %d of warp %d of block %d\n", *execute_task_id, k_max, lane, get_warp_id_in_block(), blockIdx.x);
#endif
    }
    if (lane == 0) {
        tail_by_queue_idx[k_max] = direct_start;
    }
    __syncwarp();

    #pragma unroll
    for (int kind = 0; kind < GTAP_NUM_QUEUES; ++kind) {
        int push_cnt = ctx->task_id_generated_count_by_queue_idx[kind];
        if (kind == k_max) {
            push_cnt -= *execute_task_count;
        }
        if (push_cnt <= 0) continue;

        WarpTaskQueue* q = &d_warp_task_queues[kind][warp_id_global];
        // __threadfence();
        // __syncwarp();
        if (lane == 0) {
            atomicAdd(&q->count, push_cnt);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int kind = 0; kind < GTAP_NUM_QUEUES; ++kind) {
            ctx->task_id_generated_count_by_queue_idx[kind] = 0;
        }
    }
}

// Get the current state of a task (reads from TaskHeader)
__device__ __forceinline__ int __gtap_get_task_state(int tid) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)tid;
    return 0;
#else
    return load_L2_u16t(&d_task_headers[tid].state);
#endif
}

__device__ __forceinline__ bool __gtap_set_state_for_join(int tid, int child_count, int next_state, int queue_idx_after_join) {
    if (queue_idx_after_join >= GTAP_NUM_QUEUES) {
        gtap_record_runtime_error_and_trap(
            GTAP_ERROR_INVALID_QUEUE_IDX_AFTER_JOIN, get_warp_id_global(), tid,
            queue_idx_after_join, queue_idx_after_join, GTAP_NUM_QUEUES, __LINE__);
    }
#ifndef GTAP_ASSUME_NO_TASKWAIT
    TaskHeader* hdr = &d_task_headers[tid];
#if (GTAP_NUM_QUEUES > 1)
    hdr->queue_idx = queue_idx_after_join;
#else
    (void)queue_idx_after_join;
#endif
    hdr->state = next_state;
    hdr->waiting_child_count = child_count;
#else
#if (GTAP_NUM_QUEUES > 1)
    d_task_headers[tid].queue_idx = queue_idx_after_join;
#else
    (void)tid;
    (void)queue_idx_after_join;
#endif
    (void)next_state;
#endif
    return child_count != 0;
}

// Static result retrieval stores child tids in compiler-generated task data.
// This legacy fallback should not be emitted by the current transformer.
__device__ __forceinline__ int __gtap_get_child_task_id(int parent_tid, int child_index) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)parent_tid;
    (void)child_index;
    return 0;
#else
    (void)parent_tid;
    (void)child_index;
    gtap_record_runtime_error_and_trap(
        GTAP_ERROR_INVALID_TASKWAIT, get_warp_id_global(), parent_tid, -1,
        child_index, GTAP_MAX_CHILD_TASKS, __LINE__);
    return 0;
#endif
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
#if (GTAP_NUM_QUEUES > 1)
        int parent_queue_idx = load_L2_u16t(&parent_hdr->queue_idx);
#else
        int parent_queue_idx = 0;
#endif
        reserve_unpublished_task_id(ctx, parent_queue_idx, parentId);
    }
    return rem;
}
#endif

extern "C" __device__ __forceinline__ void __gtap_finish_task(int tid, TaskContext* ctx) {
#ifdef DEBUG
    printf("finish_task: %d in lane %d of warp %d of block %d\n", tid, get_lane_id(), get_warp_id_in_block(), blockIdx.x);
#endif

#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)ctx;
    release_task_id_to_warp_pool(tid);
#else
    int lane = get_lane_id();
    TaskHeader* cached_hdr = &ctx->task_headers[lane];
    int parent_tid = cached_hdr->parent_tid;
    d_task_headers[tid].generation = cached_hdr->generation + 1;

    if (tid != 0 && load_L2_u16t(&d_task_headers[parent_tid].generation) == cached_hdr->parent_generation) {
        notify_parent(parent_tid, ctx);
        if (cached_hdr->retain_parent_result == 0) {
            release_task_id_to_warp_pool(tid);
        }
    } else {
        release_task_id_to_warp_pool(tid);
    }
#endif

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
    int child_queue_idx,
    int* out_tid,
    bool retain_parent_result
) {
#ifdef GTAP_ASSUME_NO_TASKWAIT
    (void)retain_parent_result;
#endif
    if (child_queue_idx >= GTAP_NUM_QUEUES) {
        gtap_record_runtime_error_and_trap(
            GTAP_ERROR_INVALID_QUEUE_IDX, get_warp_id_global(), self_tid,
            child_queue_idx, child_queue_idx, GTAP_NUM_QUEUES, __LINE__);
    }
    int warp_id_global = get_warp_id_global();
    TaskIdFromPool from_pool = get_task_id_from_warp_pool(&d_task_id_lists[warp_id_global], &ctx->id_list_alloc_pos, &ctx->id_list_free_pos_stale);
    int new_tid = from_pool.tid;
    if (out_tid != nullptr) {
        *out_tid = new_tid;
    }

    TaskHeader* new_hdr = &d_task_headers[new_tid];
    new_hdr->func = func;
#if (GTAP_NUM_QUEUES > 1)
    new_hdr->queue_idx = child_queue_idx;
#endif
#ifndef GTAP_ASSUME_NO_TASKWAIT
    int lane = get_lane_id();
    TaskHeader* cached_hdr = &ctx->task_headers[lane];
    new_hdr->parent_tid = self_tid;
    new_hdr->parent_generation = cached_hdr->generation;
    new_hdr->retain_parent_result = retain_parent_result ? 1 : 0;
    new_hdr->state = 0;
    new_hdr->waiting_child_count = 0;
    new_hdr->result_handle_begin = -1;
    new_hdr->result_handle_last = -1;
    new_hdr->result_handle_count = 0;
#endif

    reserve_unpublished_task_id(ctx, child_queue_idx, new_tid);

#ifndef GTAP_ASSUME_NO_TASKWAIT
    (*child_count)++;
#else
    (void)child_count;
#endif
    return __gtap_get_task_data(new_tid);
}

extern "C" __device__ __forceinline__ void __gtap_push_initial_task(
    void (*func)(void*, int, TaskContext*),
    int initial_queue_idx
) {
    int warp_id_global = get_warp_id_global();
    int new_tid = 0;

    TaskHeader* initial_hdr = &d_task_headers[new_tid];
    initial_hdr->func = func;
#if (GTAP_NUM_QUEUES > 1)
    initial_hdr->queue_idx = initial_queue_idx;
#endif
#ifndef GTAP_ASSUME_NO_TASKWAIT
    initial_hdr->state = 0;
    initial_hdr->retain_parent_result = 0;
    initial_hdr->parent_tid = 0;
    initial_hdr->parent_generation = 0;
    initial_hdr->waiting_child_count = 0;
    initial_hdr->result_handle_begin = -1;
    initial_hdr->result_handle_last = -1;
    initial_hdr->result_handle_count = 0;
#endif

    // Task data is copied from the compiler-generated code (out of this function)

    WarpTaskQueue* wq = &d_warp_task_queues[initial_queue_idx][warp_id_global];
    wq->queue[0] = new_tid;
    __threadfence();
    // atomicExch(&d_active_worker_count, 1);
}

template<TerminationMode M>
__device__ __forceinline__ void __gtap_execute_task_loop_device_impl() {
    const int warp_id_in_block = get_warp_id_in_block();
    const int warp_id_global = get_warp_id_global();
    const int lane = get_lane_id();

    int execute_task_id = 0;
    int execute_task_count = 0;
    bool prev_get_task = (warp_id_global == 0);
    bool should_continue = true;

    __shared__ TaskContext warp_contexts[GTAP_NUM_WARPS];
    __shared__ int tail_by_queue_idx[GTAP_NUM_WARPS][GTAP_NUM_QUEUES];

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
        warp_contexts[warp_id_in_block].tail_by_queue_idx = tail_by_queue_idx[warp_id_in_block];
        warp_contexts[warp_id_in_block].id_list_free_pos_stale = GTAP_TOTAL_TASK_IDS_PER_WARP;
        #pragma unroll
        for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
            warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
            tail_by_queue_idx[warp_id_in_block][k] = 0;
        }
        if (warp_id_global == 0) {
#ifdef PROFILE
            having_task_time[warp_id_global][0] = get_global_time();
#endif
            warp_contexts[0].id_list_alloc_pos = 1;
            WarpTaskQueue* q = &d_warp_task_queues[0][0];
            store_L2(&q->count, 1);
            tail_by_queue_idx[0][0] = 1;
        } else {
            warp_contexts[warp_id_in_block].id_list_alloc_pos = 0;
        }
    }
    __syncwarp();

    while (should_continue) {
        if (execute_task_count == 0) {
#if GTAP_NUM_QUEUES > 1
            #pragma unroll
            for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
                if (prev_get_task && execute_task_count < GTAP_WARP_SIZE) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int pop_count = pop_batch(&execute_task_id, remaining, &tail_by_queue_idx[warp_id_in_block][warp_contexts[warp_id_in_block].queue_idx], warp_contexts[warp_id_in_block].queue_idx);
                    execute_task_count += pop_count;
                }
                if (execute_task_count < GTAP_WARP_SIZE) {
                    int remaining = GTAP_WARP_SIZE - execute_task_count;
                    int steal_count = steal_batch<M>(&execute_task_id, remaining, warp_contexts[warp_id_in_block].queue_idx, prev_get_task);
                    execute_task_count += steal_count;
                }
                if (execute_task_count != 0) break;
                warp_contexts[warp_id_in_block].queue_idx = (warp_contexts[warp_id_in_block].queue_idx + 1) % GTAP_NUM_QUEUES;
            }
#else
            if (prev_get_task && execute_task_count < GTAP_WARP_SIZE) {
                int remaining = GTAP_WARP_SIZE - execute_task_count;
                int pop_count = pop_batch(&execute_task_id, remaining, &tail_by_queue_idx[warp_id_in_block][0], 0);
                execute_task_count += pop_count;
            }
            if (execute_task_count < GTAP_WARP_SIZE) {
                int remaining = GTAP_WARP_SIZE - execute_task_count;
                int steal_count = steal_batch<M>(&execute_task_id, remaining, 0, prev_get_task);
                execute_task_count += steal_count;
            }
#endif
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
                                if (d_warp_task_queues[k][warp_id_global].queue_head < tail_by_queue_idx[warp_id_in_block][k]) {
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
#if (GTAP_NUM_QUEUES > 1)
                #pragma unroll
                for (int k = 0; k < GTAP_NUM_QUEUES; ++k) {
                    warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[k] = 0;
                }
#else
                warp_contexts[warp_id_in_block].task_id_generated_count_by_queue_idx[0] = 0;
#endif
            }
            __syncwarp();
        }

        if (lane < execute_task_count) {
            // Copy task header to TaskContext for reuse in task function (using L2 load)
#ifndef GTAP_ASSUME_NO_TASKWAIT
            {
                TaskHeader* src_hdr = &d_task_headers[execute_task_id];
                TaskHeader* dst_hdr = &warp_contexts[warp_id_in_block].task_headers[lane];
                dst_hdr->generation = load_L2_u16t(&src_hdr->generation);
                dst_hdr->retain_parent_result = load_L2_u16t(&src_hdr->retain_parent_result);
                dst_hdr->parent_tid = load_L2(&src_hdr->parent_tid);
                dst_hdr->parent_generation = load_L2_u16t(&src_hdr->parent_generation);
            }
#endif

#ifdef PROFILE
            if (lane == 0) {
                if (working_time_idx[warp_id_in_block] < MAX_PROFILE_DATA) {
                    working_time[warp_id_global][working_time_idx[warp_id_in_block]] = get_global_time();
                    tasks_processed_count[warp_id_global][working_time_idx[warp_id_in_block]] = execute_task_count;
                    working_time_idx[warp_id_in_block]++;
                }
            }
#endif
            void* task_data = __gtap_get_task_data(execute_task_id);
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

        push_batch(
            &warp_contexts[warp_id_in_block], &execute_task_id,
            &execute_task_count, tail_by_queue_idx[warp_id_in_block]
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
