#pragma once

#include "config_defaults.cuh"

// Worker scratch. Steal copies task args (slot, prefix, range), not these
// buffers. A stolen task rebuilds bitsets from adjacency + prefix.
//
// cand (g_cand_scratch)
//   Per worker: int[g_max_candidates]. Sorted vertex IDs N+(u)∩N+(v).
//   Filled in count_from_edge / count_candidate_range_from_edge, then used
//   to build an adjacency matrix. Child slot-tasks do not keep this array.
//
// bitset (g_bit_scratch)
//   Per worker: g_bit_buffer_count sets, each max_words() uint64s.
//   Layout [worker][buffer_index][word]. Bits are candidate *indices*
//   (rows of the adjacency), not vertex IDs.
//   Orientation: count = k. count_cliques_in_bitset keeps the parent set
//   at buffer_index and writes the child at buffer_index+1, so one slot
//   per live DFS frame. Restore uses [0, depth] as that chain.
//   Pivot: count = 2 * measured max; even = PL, odd = CL, per level.
//
// adjacency (g_candidate_adjacency_scratch)
//   cand×cand bit-matrix, pitch max_words() per row.
//   [0, GTAP_ADJACENCY_SLOTS) : shared. Heavy edge takes a slot, children
//   reuse it, parent taskwaits then releases.
//   [SLOTS, SLOTS + num_workers) : one private matrix per worker.
//   Light path uses this and does not spawn. Heavy + no free slot also
//   uses this, but still spawns from_edge (each child rebuilds).
//
// pivot stack (g_pivot_stack_scratch)
//   Pivot only. Allocated in alloc_scratch when BitScratchKind::ByMaxCandidates.
//   Accessors: pivot/pivot_stack.cuh.

__device__ __constant__ const int* g_row_ptr;
__device__ __constant__ const int* g_col_idx;
__device__ __constant__ const int* g_edge_src;
__device__ __constant__ int* g_cand_scratch;
__device__ __constant__ unsigned long long* g_bit_scratch;
__device__ __constant__ unsigned long long* g_candidate_adjacency_scratch;
__device__ __constant__ int* g_candidate_adjacency_slot_stack;
__device__ __constant__ int* g_pivot_stack_scratch;
__device__ int g_candidate_adjacency_slot_top;
__device__ int g_candidate_adjacency_slot_lock;
__device__ unsigned long long g_answer;
__device__ int g_scratch_overflow;
#ifdef GTAP_STATS
__device__ int g_candidate_highwater;
__device__ unsigned long long g_task_max_cycles[K_PROFILE_KINDS];
__device__ int g_task_max_info[K_PROFILE_KINDS * K_PROFILE_FIELDS];
#endif
__device__ __constant__ int g_num_edges;
__device__ __constant__ int g_clique_k;
__device__ __constant__ int g_max_candidates;
__device__ __constant__ int g_bit_buffer_count;
__device__ __constant__ int g_num_workers;

static inline cudaError_t bind_graph(
        const int* d_row_ptr, const int* d_col_idx, const int* d_edge_src,
        int* d_cand_scratch, unsigned long long* d_bit_scratch,
        unsigned long long* d_candidate_adjacency_scratch,
        int* d_candidate_adjacency_slot_stack) {
    cudaError_t st = cudaMemcpyToSymbol(g_row_ptr, &d_row_ptr, sizeof(d_row_ptr));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_col_idx, &d_col_idx, sizeof(d_col_idx));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_edge_src, &d_edge_src, sizeof(d_edge_src));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_cand_scratch, &d_cand_scratch, sizeof(d_cand_scratch));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(g_bit_scratch, &d_bit_scratch, sizeof(d_bit_scratch));
    if (st != cudaSuccess) return st;
    st = cudaMemcpyToSymbol(
        g_candidate_adjacency_scratch,
        &d_candidate_adjacency_scratch,
        sizeof(d_candidate_adjacency_scratch));
    if (st != cudaSuccess) return st;
    return cudaMemcpyToSymbol(
        g_candidate_adjacency_slot_stack, &d_candidate_adjacency_slot_stack, sizeof(d_candidate_adjacency_slot_stack));
}

static inline cudaError_t bind_pivot_stack(int* d_pivot_stack) {
    return cudaMemcpyToSymbol(
        g_pivot_stack_scratch, &d_pivot_stack, sizeof(d_pivot_stack));
}

__device__ __forceinline__ int worker_id_global() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int max_words() {
    return (g_max_candidates + 63) >> 6;
}

__device__ __forceinline__ int words_for_len(int len) {
    return (len + 63) >> 6;
}

// This worker's vertex-ID list for the current edge.
__device__ __forceinline__ int* cand_buffer() {
    return g_cand_scratch + (size_t)worker_id_global() * g_max_candidates;
}

// This worker's selectable-index set at `buffer_index` (search depth).
__device__ __forceinline__ unsigned long long* candidate_bitset_buffer(
        int buffer_index) {
    int words = max_words();
    return g_bit_scratch +
           ((size_t)worker_id_global() * g_bit_buffer_count + buffer_index) *
           (size_t)words;
}

// This worker's private adjacency matrix (after the shared slot block).
__device__ __forceinline__ unsigned long long* candidate_adjacency_buffer() {
    return g_candidate_adjacency_scratch +
           ((size_t)GTAP_ADJACENCY_SLOTS + (size_t)worker_id_global()) *
           (size_t)g_max_candidates *
           (size_t)max_words();
}

// Shared adjacency matrix at `slot`. Children of a heavy edge reuse this.
__device__ __forceinline__ unsigned long long* candidate_adjacency_slot_buffer(int slot) {
    return g_candidate_adjacency_scratch +
           (size_t)slot *
           (size_t)g_max_candidates *
           (size_t)max_words();
}

__device__ __forceinline__ void mark_overflow() {
    atomicExch(&g_scratch_overflow, 1);
}

#ifdef GTAP_STATS
__device__ __forceinline__ unsigned long long global_time() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

__device__ __forceinline__ void candidate_adjacency_slot_lock() {
    while (atomicCAS(&g_candidate_adjacency_slot_lock, 0, 1) != 0) {
    }
}

__device__ __forceinline__ void candidate_adjacency_slot_unlock() {
    __threadfence();
    atomicExch(&g_candidate_adjacency_slot_lock, 0);
}

__device__ __forceinline__ int acquire_candidate_adjacency_slot() {
    candidate_adjacency_slot_lock();
    if (g_candidate_adjacency_slot_top <= 0) {
        candidate_adjacency_slot_unlock();
        return -1;
    }
    --g_candidate_adjacency_slot_top;
    int slot = g_candidate_adjacency_slot_stack[g_candidate_adjacency_slot_top];
    candidate_adjacency_slot_unlock();
    return slot;
}

__device__ __forceinline__ void release_candidate_adjacency_slot(int slot) {
    candidate_adjacency_slot_lock();
    if (g_candidate_adjacency_slot_top >= GTAP_ADJACENCY_SLOTS) {
        candidate_adjacency_slot_unlock();
        mark_overflow();
        return;
    }
    g_candidate_adjacency_slot_stack[g_candidate_adjacency_slot_top] = slot;
    ++g_candidate_adjacency_slot_top;
    candidate_adjacency_slot_unlock();
}

#ifdef GTAP_STATS
__device__ __forceinline__ void record_task_profile(
        int kind, unsigned long long cycles, int u, int v, int cand_count,
        int a, int b, int c, int d) {
    unsigned long long old = atomicMax(&g_task_max_cycles[kind], cycles);
    if (cycles > old) {
        int base = kind * K_PROFILE_FIELDS;
        g_task_max_info[base + 0] = u;
        g_task_max_info[base + 1] = v;
        g_task_max_info[base + 2] = cand_count;
        g_task_max_info[base + 3] = a;
        g_task_max_info[base + 4] = b;
        g_task_max_info[base + 5] = c;
        g_task_max_info[base + 6] = d;
        g_task_max_info[base + 7] = worker_id_global();
    }
}
#endif
