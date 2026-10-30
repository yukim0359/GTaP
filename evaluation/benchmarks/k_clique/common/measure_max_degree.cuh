#pragma once

#include <cuda_runtime.h>

namespace {

constexpr int kEdgeDegreeBlock = 128;

// Same reduction as KCGPU getEdgeDegree_kernel: max out-degree of either
// endpoint of an oriented edge. The per-edge edgePtr store is omitted;
// GTaP does not bucket by edge degree.
template <int BLOCK_DIM_X>
__global__ void get_edge_degree(
        const int* row_ptr, const int* col_idx, const int* edge_src,
        int num_edges, int* max_degree) {
    const int gtid = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
    __shared__ int smem[BLOCK_DIM_X];
    int degree = 0;
    if (gtid < num_edges) {
        int src = edge_src[gtid];
        int dst = col_idx[gtid];
        int src_deg = row_ptr[src + 1] - row_ptr[src];
        int dst_deg = row_ptr[dst + 1] - row_ptr[dst];
        degree = src_deg > dst_deg ? src_deg : dst_deg;
    }
    smem[threadIdx.x] = degree;
    __syncthreads();
    for (unsigned stride = BLOCK_DIM_X / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            int other = smem[threadIdx.x + stride];
            if (other > smem[threadIdx.x]) smem[threadIdx.x] = other;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(max_degree, smem[0]);
}

}  // namespace

static cudaError_t measure_max_endpoint_degree(
        const int* d_row_ptr, const int* d_col_idx, const int* d_edge_src,
        int num_edges, int* out_max_degree) {
    if (out_max_degree == nullptr) return cudaErrorInvalidValue;
    *out_max_degree = 0;
    if (num_edges <= 0) return cudaSuccess;
    if (d_row_ptr == nullptr || d_col_idx == nullptr || d_edge_src == nullptr) {
        return cudaErrorInvalidValue;
    }

    int* d_max = nullptr;
    cudaError_t err = cudaMalloc(&d_max, sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMemset(d_max, 0, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_max);
        return err;
    }

    const int block = kEdgeDegreeBlock;
    const int grid = (num_edges + block - 1) / block;
    get_edge_degree<kEdgeDegreeBlock><<<grid, block>>>(
            d_row_ptr, d_col_idx, d_edge_src, num_edges, d_max);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        err = cudaMemcpy(out_max_degree, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_max);
    return err;
}
