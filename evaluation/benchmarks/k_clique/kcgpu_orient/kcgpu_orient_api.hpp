#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// Graph orientation modes matching KCGPU (main.cu Degree / Degeneracy).
enum KcgpuOrientMode {
    KCGPU_ORIENT_DEGREE = 0,
    KCGPU_ORIENT_DEGEN = 1,
};

// Pinned-host CSR + device rowPtr, matching KCGPU main.cu graph layout before preprocess.
struct KcgpuOrientHostState {
    unsigned int* h_row_ptr = nullptr;
    unsigned int* h_row_ind = nullptr;
    unsigned int* h_col_ind = nullptr;
    unsigned int* d_row_ptr = nullptr;
    int n = 0;
    unsigned int m = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Allocate pinned CSR and build row_ind on the host (KCGPU "read graph" CSR layout).
cudaError_t kcgpu_orient_host_state_create(
    const int* h_undirected_row_ptr,
    const int* h_undirected_col_idx,
    int n,
    KcgpuOrientHostState** out_state);

// Transfer rowPtr to the device (KCGPU Transfer Time).
cudaError_t kcgpu_orient_transfer(KcgpuOrientHostState* state, int device_id);

// Run orientation kernels only (KCGPU Preprocess time). row/col stay on pinned host.
cudaError_t kcgpu_orient_preprocess(
    KcgpuOrientHostState* state,
    int mode,
    int device_id,
    int** d_out_row_ptr,
    int** d_out_col_idx,
    size_t* out_num_edges);

void kcgpu_orient_host_state_destroy(KcgpuOrientHostState* state);

#ifdef __cplusplus
}
#endif
