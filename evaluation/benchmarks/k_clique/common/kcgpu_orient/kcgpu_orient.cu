// KCGPU graph-orientation pipeline extracted for shared use by GTaP benchmarks.
// Data layout and kernels mirror KCGPU/src/main.cu (Degree / Degeneracy orientation).

#include "kcgpu_orient_api.hpp"

#include <cstdio>
#include <cstdlib>

#include "Logger.cuh"
#include "CGArray.cuh"
#include "TriCountPrim.cuh"
#include "defs.cuh"
#include "main_support.cuh"
#include "kcore.cuh"

namespace {

using T = uint;

static void build_row_ind_host(const T* row_ptr, T* row_ind, T n) {
    for (T u = 0; u < n; ++u) {
        for (T ei = row_ptr[u]; ei < row_ptr[u + 1]; ++ei) {
            row_ind[ei] = u;
        }
    }
}

}  // namespace

extern "C" cudaError_t kcgpu_orient_host_state_create(
    const int* h_undirected_row_ptr,
    const int* h_undirected_col_idx,
    int n,
    KcgpuOrientHostState** out_state) {
    if (h_undirected_row_ptr == nullptr || h_undirected_col_idx == nullptr || out_state == nullptr || n <= 0) {
        return cudaErrorInvalidValue;
    }

    const T num_nodes = static_cast<T>(n);
    const T m = static_cast<T>(h_undirected_row_ptr[n]);
    if (m <= 0) return cudaErrorInvalidValue;

    auto* state = static_cast<KcgpuOrientHostState*>(std::calloc(1, sizeof(KcgpuOrientHostState)));
    if (state == nullptr) return cudaErrorMemoryAllocation;

    state->n = n;
    state->m = m;

    cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&state->h_row_ptr),
        (size_t)(num_nodes + 1) * sizeof(unsigned int));
    if (err != cudaSuccess) goto fail;
    err = cudaMallocHost(reinterpret_cast<void**>(&state->h_row_ind), (size_t)m * sizeof(unsigned int));
    if (err != cudaSuccess) goto fail;
    err = cudaMallocHost(reinterpret_cast<void**>(&state->h_col_ind), (size_t)m * sizeof(unsigned int));
    if (err != cudaSuccess) goto fail;

    for (T i = 0; i <= num_nodes; ++i) {
        state->h_row_ptr[i] = static_cast<unsigned int>(h_undirected_row_ptr[i]);
    }
    for (T i = 0; i < m; ++i) {
        state->h_col_ind[i] = static_cast<unsigned int>(h_undirected_col_idx[i]);
    }
    build_row_ind_host(state->h_row_ptr, state->h_row_ind, num_nodes);

    *out_state = state;
    return cudaSuccess;

fail:
    kcgpu_orient_host_state_destroy(state);
    return err;
}

extern "C" cudaError_t kcgpu_orient_transfer(KcgpuOrientHostState* state, int device_id) {
    if (state == nullptr || state->h_row_ptr == nullptr || state->n <= 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return err;

    if (state->d_row_ptr != nullptr) {
        cudaFree(state->d_row_ptr);
        state->d_row_ptr = nullptr;
    }

    const T num_nodes = static_cast<T>(state->n);
    err = cudaMalloc(reinterpret_cast<void**>(&state->d_row_ptr),
        (size_t)(num_nodes + 1) * sizeof(unsigned int));
    if (err != cudaSuccess) return err;

    return cudaMemcpy(
        state->d_row_ptr,
        state->h_row_ptr,
        (size_t)(num_nodes + 1) * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
}

extern "C" cudaError_t kcgpu_orient_preprocess(
    KcgpuOrientHostState* state,
    int mode,
    int device_id,
    int** d_out_row_ptr,
    int** d_out_col_idx,
    size_t* out_num_edges) {
    if (state == nullptr || state->d_row_ptr == nullptr || state->h_row_ind == nullptr ||
        state->h_col_ind == nullptr || d_out_row_ptr == nullptr || d_out_col_idx == nullptr ||
        out_num_edges == nullptr) {
        return cudaErrorInvalidValue;
    }

    *d_out_row_ptr = nullptr;
    *d_out_col_idx = nullptr;
    *out_num_edges = 0;

    const T num_nodes = static_cast<T>(state->n);
    const T m = static_cast<T>(state->m);
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return err;

    graph::COOCSRGraph_d<T> gd{};
    gd.numNodes = num_nodes;
    gd.numEdges = m;
    gd.capacity = m;
    gd.rowPtr = state->d_row_ptr;
    gd.rowInd = state->h_row_ind;
    gd.colInd = state->h_col_ind;

    graph::SingleGPU_Kcore<T, PeelType> mohacore(device_id);
    if (mode == KCGPU_ORIENT_DEGEN) {
        mohacore.findKcoreIncremental_async(3, 1000, gd, 0, 0);
    } else {
        mohacore.getNodeDegree(gd);
    }

    const AllocationTypeEnum half_alloc = AllocationTypeEnum::gpu;
    graph::GPUArray<T> row_ind_half("Half Row Index", half_alloc, m / 2, device_id);
    graph::GPUArray<T> col_ind_half("Half Col Index", half_alloc, m / 2, device_id);
    graph::GPUArray<T> new_row_ptr("New Row Pointer", half_alloc, num_nodes + 1, device_id);
    graph::GPUArray<T> asc("ASC temp", AllocationTypeEnum::unified, m, device_id);
    graph::GPUArray<bool> keep("Keep temp", AllocationTypeEnum::unified, m, device_id);

    if (mode == KCGPU_ORIENT_DEGREE) {
        execKernel(
            (init<T, PeelType>),
            ((m - 1) / 51200) + 1,
            512,
            device_id,
            false,
            gd,
            asc.gdata(),
            keep.gdata(),
            mohacore.nodeDegree.gdata());
    } else {
        execKernel(
            (init<T, PeelType>),
            ((m - 1) / 51200) + 1,
            512,
            device_id,
            false,
            gd,
            asc.gdata(),
            keep.gdata(),
            mohacore.nodeDegree.gdata(),
            mohacore.nodePriority.gdata());
    }

    const uint32_t new_num_edges =
        CUBSelect(gd.rowInd, row_ind_half.gdata(), keep.gdata(), m, device_id);
    const uint32_t new_num_edges_col =
        CUBSelect(gd.colInd, col_ind_half.gdata(), keep.gdata(), m, device_id);
    if (new_num_edges != new_num_edges_col || new_num_edges == 0) {
        return cudaErrorUnknown;
    }

    execKernel(
        (warp_detect_deleted_edges<T>),
        (32 * num_nodes + 128 - 1) / 128,
        128,
        device_id,
        false,
        gd.rowPtr,
        num_nodes,
        keep.gdata(),
        new_row_ptr.gdata());

    const T total = CUBScanExclusive<T, T>(
        new_row_ptr.gdata(), new_row_ptr.gdata(), num_nodes, device_id, 0, half_alloc);
    new_row_ptr.setSingle(num_nodes, total, false);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return err;

    if (total != new_num_edges) {
        fprintf(stderr,
                "kcgpu_orient: oriented CSR size mismatch (%u vs %u)\n",
                static_cast<unsigned>(total),
                static_cast<unsigned>(new_num_edges));
        return cudaErrorUnknown;
    }

    err = cudaMalloc(reinterpret_cast<void**>(d_out_row_ptr), (size_t)(num_nodes + 1) * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(reinterpret_cast<void**>(d_out_col_idx), (size_t)new_num_edges * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(*d_out_row_ptr);
        *d_out_row_ptr = nullptr;
        return err;
    }

    err = cudaMemcpy(
        *d_out_row_ptr,
        new_row_ptr.gdata(),
        (size_t)(num_nodes + 1) * sizeof(T),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto fail_output;
    err = cudaMemcpy(
        *d_out_col_idx,
        col_ind_half.gdata(),
        (size_t)new_num_edges * sizeof(T),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) goto fail_output;

    *out_num_edges = static_cast<size_t>(new_num_edges);
    return cudaSuccess;

fail_output:
    if (*d_out_row_ptr != nullptr) cudaFree(*d_out_row_ptr);
    if (*d_out_col_idx != nullptr) cudaFree(*d_out_col_idx);
    *d_out_row_ptr = nullptr;
    *d_out_col_idx = nullptr;
    return err;
}

extern "C" void kcgpu_orient_host_state_destroy(KcgpuOrientHostState* state) {
    if (state == nullptr) return;
    if (state->h_row_ptr != nullptr) cudaFreeHost(state->h_row_ptr);
    if (state->h_row_ind != nullptr) cudaFreeHost(state->h_row_ind);
    if (state->h_col_ind != nullptr) cudaFreeHost(state->h_col_ind);
    if (state->d_row_ptr != nullptr) cudaFree(state->d_row_ptr);
    std::free(state);
}
