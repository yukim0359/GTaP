#include "k_clique_gpu_orient.cuh"

#include "kcgpu_orient/kcgpu_orient_api.hpp"

#include "k_clique_host_graph.hpp"

#include <cstdio>
#include <time.h>

namespace {

constexpr int kBlockSize = 256;

__global__ void build_edge_src_kernel(const int* __restrict__ row_ptr, int* __restrict__ edge_src, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    for (int ei = row_ptr[u]; ei < row_ptr[u + 1]; ++ei) {
        edge_src[ei] = u;
    }
}

static int to_kcgpu_mode(GtapOrientMode mode) {
    return mode == GTAP_ORIENT_DEGEN ? KCGPU_ORIENT_DEGEN : KCGPU_ORIENT_DEGREE;
}

}  // namespace

cudaError_t gtap_gpu_orient_csr(
    const int* h_undirected_row_ptr,
    const int* h_undirected_col_idx,
    int n,
    size_t undirected_edges,
    GtapOrientMode mode,
    int** d_out_row_ptr,
    int** d_out_col_idx,
    int** d_out_edge_src,
    size_t* out_num_edges,
    HostGraph* h_out_exec_graph,
    double* out_transfer_ms,
    double* out_orient_ms) {
    if (d_out_row_ptr == nullptr || d_out_col_idx == nullptr || d_out_edge_src == nullptr ||
        out_num_edges == nullptr || h_undirected_row_ptr == nullptr || h_undirected_col_idx == nullptr ||
        n <= 0) {
        return cudaErrorInvalidValue;
    }

    *d_out_row_ptr = nullptr;
    *d_out_col_idx = nullptr;
    *d_out_edge_src = nullptr;
    *out_num_edges = 0;
    if (out_transfer_ms != nullptr) *out_transfer_ms = 0.0;
    if (out_orient_ms != nullptr) *out_orient_ms = 0.0;

    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) return err;

    KcgpuOrientHostState* host_state = nullptr;
    err = kcgpu_orient_host_state_create(h_undirected_row_ptr, h_undirected_col_idx, n, &host_state);
    if (err != cudaSuccess) return err;

    timespec transfer_start, transfer_stop, orient_start, orient_stop;
    clock_gettime(CLOCK_MONOTONIC, &transfer_start);
    err = kcgpu_orient_transfer(host_state, device_id);
    clock_gettime(CLOCK_MONOTONIC, &transfer_stop);
    if (err != cudaSuccess) {
        kcgpu_orient_host_state_destroy(host_state);
        return err;
    }

    size_t num_oriented_edges = 0;
    clock_gettime(CLOCK_MONOTONIC, &orient_start);
    err = kcgpu_orient_preprocess(
        host_state,
        to_kcgpu_mode(mode),
        device_id,
        d_out_row_ptr,
        d_out_col_idx,
        &num_oriented_edges);
    if (err != cudaSuccess) {
        kcgpu_orient_host_state_destroy(host_state);
        return err;
    }

    err = cudaMalloc(d_out_edge_src, num_oriented_edges * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(*d_out_row_ptr);
        cudaFree(*d_out_col_idx);
        *d_out_row_ptr = nullptr;
        *d_out_col_idx = nullptr;
        kcgpu_orient_host_state_destroy(host_state);
        return err;
    }

    {
        const int grid = (n + kBlockSize - 1) / kBlockSize;
        build_edge_src_kernel<<<grid, kBlockSize>>>(*d_out_row_ptr, *d_out_edge_src, n);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;

    clock_gettime(CLOCK_MONOTONIC, &orient_stop);
    kcgpu_orient_host_state_destroy(host_state);

    *out_num_edges = num_oriented_edges;
    if (out_transfer_ms != nullptr) {
        *out_transfer_ms = elapsed_ms(transfer_start, transfer_stop);
    }
    if (out_orient_ms != nullptr) {
        *out_orient_ms = elapsed_ms(orient_start, orient_stop);
    }

    if (h_out_exec_graph != nullptr) {
        h_out_exec_graph->n = n;
        h_out_exec_graph->undirected_edges = undirected_edges;
        h_out_exec_graph->oriented = true;
        h_out_exec_graph->row_ptr.resize((size_t)n + 1);
        h_out_exec_graph->col_idx.resize(num_oriented_edges);
        err = cudaMemcpy(
            h_out_exec_graph->row_ptr.data(),
            *d_out_row_ptr,
            (size_t)(n + 1) * sizeof(int),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(
            h_out_exec_graph->col_idx.data(),
            *d_out_col_idx,
            num_oriented_edges * sizeof(int),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) goto fail;
    }

    return cudaSuccess;

fail:
    cudaFree(*d_out_row_ptr);
    cudaFree(*d_out_col_idx);
    cudaFree(*d_out_edge_src);
    *d_out_row_ptr = nullptr;
    *d_out_col_idx = nullptr;
    *d_out_edge_src = nullptr;
    kcgpu_orient_host_state_destroy(host_state);
    return err;
}
