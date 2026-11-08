#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef GQ
#include "experimental/gtap_thread_gq.cuh"
#else
#ifdef CHASELEV
#include "experimental/gtap_thread_chaselev.cuh"
#else
#include "gtap_thread.cuh"
#endif
#endif
#include "../../common/gtap_launch.hpp"

#ifndef GTAP_SM_COUNT
#define GTAP_SM_COUNT 0
#endif

__device__ int d_result;

#pragma gtap function
__device__ int fib(int n) {
    if (n < 2) {
        return n;
    }
    int a, b;
    #pragma gtap task
    a = fib(n - 1);
    #pragma gtap task
    b = fib(n - 2);
    #pragma gtap taskwait
    return a + b;
}

__global__ void exec_kernel(int n) {
    #pragma gtap entry
    d_result = fib(n);
}

static bool check_cu(CUresult status, const char *call) {
    if (status == CUDA_SUCCESS) return true;
    const char *name = "unknown";
    cuGetErrorName(status, &name);
    fprintf(stderr, "%s failed: %s\n", call, name);
    return false;
}

// requested == 0 keeps the full chip. Otherwise the primary context is replaced
// with a green context of that many SMs before any CUDA runtime call.
static bool limit_streaming_multiprocessors(unsigned int requested,
                                            unsigned int *provisioned) {
    if (!check_cu(cuInit(0), "cuInit")) return false;
    CUdevice device;
    if (!check_cu(cuDeviceGet(&device, 0), "cuDeviceGet")) return false;

    int device_sms = 0;
    if (!check_cu(cuDeviceGetAttribute(
            &device_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
            "cuDeviceGetAttribute")) {
        return false;
    }
    CUdevResource resource = {};
    if (!check_cu(cuDeviceGetDevResource(
            device, &resource, CU_DEV_RESOURCE_TYPE_SM),
            "cuDeviceGetDevResource")) {
        return false;
    }
    printf("device_sms=%d resource_sms=%u min_partition=%u alignment=%u\n",
           device_sms, resource.sm.smCount, resource.sm.minSmPartitionSize,
           resource.sm.smCoscheduledAlignment);
    fflush(stdout);

    if (requested == 0) {
        *provisioned = static_cast<unsigned int>(device_sms);
        return true;
    }

    unsigned int groups = 1;
    CUdevResource partition = {};
    CUdevResource remaining = {};
    if (!check_cu(cuDevSmResourceSplitByCount(
            &partition, &groups, &resource, &remaining, 0, requested),
            "cuDevSmResourceSplitByCount")) {
        return false;
    }
    if (groups != 1) {
        fprintf(stderr, "expected one SM partition, got %u\n", groups);
        return false;
    }
    *provisioned = partition.sm.smCount;

    CUdevResourceDesc descriptor;
    if (!check_cu(cuDevResourceGenerateDesc(&descriptor, &partition, 1),
                  "cuDevResourceGenerateDesc")) {
        return false;
    }
    CUgreenCtx green;
    if (!check_cu(cuGreenCtxCreate(
            &green, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM),
            "cuGreenCtxCreate")) {
        return false;
    }
    CUcontext context;
    if (!check_cu(cuCtxFromGreenCtx(&context, green), "cuCtxFromGreenCtx")) {
        return false;
    }
    return check_cu(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

__global__ void record_smid(int *smids) {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    if (threadIdx.x == 0) smids[blockIdx.x] = static_cast<int>(smid);
}

static int observe_sm_count(int grid, int block) {
    int *device_smids = nullptr;
    cudaError_t status = cudaMalloc(&device_smids, sizeof(int) * static_cast<size_t>(grid));
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(smids) failed: %s\n", cudaGetErrorString(status));
        return -1;
    }
    record_smid<<<grid, block>>>(device_smids);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "smid probe failed: %s\n", cudaGetErrorString(status));
        cudaFree(device_smids);
        return -1;
    }
    int *host_smids = (int *)malloc(sizeof(int) * (size_t)grid);
    if (host_smids == NULL) {
        fprintf(stderr, "malloc(smids) failed\n");
        cudaFree(device_smids);
        return -1;
    }
    status = cudaMemcpy(host_smids, device_smids, sizeof(int) * (size_t)grid,
                        cudaMemcpyDeviceToHost);
    cudaFree(device_smids);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(smids) failed: %s\n", cudaGetErrorString(status));
        free(host_smids);
        return -1;
    }

    int seen[512] = {};
    int unique = 0;
    for (int i = 0; i < grid; ++i) {
        int smid = host_smids[i];
        if (smid < 0 || smid >= 512) {
            fprintf(stderr, "smid %d is outside 0..511\n", smid);
            free(host_smids);
            return -1;
        }
        if (!seen[smid]) {
            seen[smid] = 1;
            ++unique;
        }
    }
    free(host_smids);
    return unique;
}

int main(int argc, char** argv) {
    int n = 40;
    if (argc >= 2) n = atoi(argv[1]);

    unsigned int provisioned_sms = 0;
    if (!limit_streaming_multiprocessors(GTAP_SM_COUNT, &provisioned_sms)) {
        return 1;
    }

    gtap_thread_config config{
        .grid_size = GTAP_GRID_SIZE,
        .block_size = GTAP_BLOCK_SIZE,
        .max_tasks_per_warp = GTAP_MAX_TASKS_PER_WARP,
        .num_queues = 1,
    };
    if (!apply_thread_launch_env(config.grid_size, config.block_size, config.max_tasks_per_warp)) {
        return 1;
    }
    cudaError_t err = gtap_initialize(config);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int observed_sms = observe_sm_count(config.grid_size, config.block_size);
    printf("SM scaling: requested=%u provisioned=%u observed=%d grid=%d\n",
           static_cast<unsigned int>(GTAP_SM_COUNT), provisioned_sms,
           observed_sms, config.grid_size);
    fflush(stdout);
    if (observed_sms != static_cast<int>(provisioned_sms) ||
        (GTAP_SM_COUNT != 0 &&
         provisioned_sms != static_cast<unsigned int>(GTAP_SM_COUNT))) {
        fprintf(stderr, "error: observed SM count does not match the partition\n");
        gtap_finalize();
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    err = gtap_launch(exec_kernel, n);
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
        gtap_finalize();
        return 1;
    }
    cudaEventRecord(stop);
    err = gtap_synchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        gtap_finalize();
        return 1;
    }
    cudaEventSynchronize(stop);
    
    int h_result;
    cudaMemcpyFromSymbol(&h_result, d_result, sizeof(int));
    printf("Fibonacci of %d is %d\n", n, h_result);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Execution time: %.3f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
