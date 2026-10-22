#pragma once

#include <cstddef>
#include <cuda_runtime.h>

enum GtapOrientMode {
    GTAP_ORIENT_DEGREE = 0,
    GTAP_ORIENT_DEGEN = 1,
};

static const char* gtap_orient_mode_name(GtapOrientMode mode) {
    switch (mode) {
        case GTAP_ORIENT_DEGEN: return "degen";
        case GTAP_ORIENT_DEGREE:
        default: return "degree";
    }
}

// Orient an undirected CSR on the GPU using KCGPU orientation kernels (kcgpu_orient/).
// Timing matches KCGPU: transfer = rowPtr H2D; orient = preprocess kernels (+ edge_src).
// Outputs device CSR + edge_src. Caller owns *d_out_* (cudaFree).
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
    double* out_transfer_ms = nullptr,
    double* out_orient_ms = nullptr);
