#pragma once

// Per-CUDA-block scratch (blockIdx.x). All threads in a block cooperate on one task.
__device__ const int* g_row_ptr;
__device__ const int* g_col_idx;
__device__ const int* g_edge_src;
__device__ int* g_cand_scratch;
__device__ unsigned long long* g_bit_scratch;
__device__ unsigned long long* g_encode_scratch;
__device__ int* g_encode_slot_stack;
__device__ int g_encode_slot_top;
__device__ int g_encode_slot_lock;
__device__ unsigned long long g_answer;
__device__ int g_scratch_overflow;
#ifdef GTAP_K_STATS
__device__ int g_candidate_highwater;
__device__ unsigned long long g_task_max_cycles[K_PROFILE_KINDS];
__device__ int g_task_max_info[K_PROFILE_KINDS * K_PROFILE_FIELDS];
__device__ unsigned long long g_recurse_encoded_calls;
__device__ unsigned long long g_recurse_encoded_roots;
__device__ unsigned long long* g_recurse_tree_encoded;
__device__ unsigned long long g_recurse_max_encoded_tree_calls;
#endif
__device__ __constant__ int g_num_vertices;
__device__ __constant__ int g_num_edges;
__device__ __constant__ int g_clique_k;

__device__ const unsigned long long* g_nCr;

static inline cudaError_t bind_graph(
    const int* d_row_ptr,
    const int* d_col_idx,
    const int* d_edge_src,
    int* d_cand_scratch,
    unsigned long long* d_bit_scratch,
    unsigned long long* d_encode_scratch,
    int* d_encode_slot_stack) {
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
    st = cudaMemcpyToSymbol(g_encode_scratch, &d_encode_scratch, sizeof(d_encode_scratch));
    if (st != cudaSuccess) return st;
    return cudaMemcpyToSymbol(g_encode_slot_stack, &d_encode_slot_stack, sizeof(d_encode_slot_stack));
}

static inline cudaError_t bind_nCr(const unsigned long long* d_nCr) {
    return cudaMemcpyToSymbol(g_nCr, &d_nCr, sizeof(d_nCr));
}

#ifdef GTAP_K_STATS
static inline cudaError_t bind_recurse_stats(unsigned long long* d_tree_encoded) {
    return cudaMemcpyToSymbol(g_recurse_tree_encoded, &d_tree_encoded, sizeof(d_tree_encoded));
}
#endif

__device__ __forceinline__ int block_worker_id() {
    return blockIdx.x;
}

__device__ __forceinline__ int max_words() {
    return (GTAP_K_MAX_CANDIDATES + 63) >> 6;
}

__device__ __forceinline__ int words_for_len(int len) {
    return (len + 63) >> 6;
}

__device__ __forceinline__ int* cand_buffer() {
    return g_cand_scratch + (size_t)block_worker_id() * GTAP_K_MAX_CANDIDATES;
}

__device__ __forceinline__ unsigned long long* bit_buffer(int level) {
    int words = max_words();
    return g_bit_scratch +
           ((size_t)block_worker_id() * GTAP_K_BIT_BUFFER_COUNT + level) * (size_t)words;
}

__device__ __forceinline__ unsigned long long* encode_buffer() {
    return g_encode_scratch +
           ((size_t)GTAP_K_ENCODE_SLOTS + (size_t)block_worker_id()) *
           (size_t)GTAP_K_MAX_CANDIDATES *
           (size_t)max_words();
}

__device__ __forceinline__ unsigned long long* encode_slot_buffer(int slot) {
    return g_encode_scratch +
           (size_t)slot *
           (size_t)GTAP_K_MAX_CANDIDATES *
           (size_t)max_words();
}

__device__ __forceinline__ void mark_overflow() {
    atomicExch(&g_scratch_overflow, 1);
}

#ifdef GTAP_K_STATS
__device__ __forceinline__ void stats_flush_encoded_tree(int worker) {
    unsigned long long prev = g_recurse_tree_encoded[worker];
    if (prev > 0ULL) atomicMax(&g_recurse_max_encoded_tree_calls, prev);
    g_recurse_tree_encoded[worker] = 0ULL;
}

__device__ __forceinline__ void stats_on_encoded_recurse(int buf_level) {
    int worker = block_worker_id();
    atomicAdd(&g_recurse_encoded_calls, 1ULL);
    if (buf_level == 0) {
        stats_flush_encoded_tree(worker);
        g_recurse_tree_encoded[worker] = 1ULL;
        atomicAdd(&g_recurse_encoded_roots, 1ULL);
    } else {
        ++g_recurse_tree_encoded[worker];
    }
}

__device__ __forceinline__ unsigned long long global_time() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

__device__ __forceinline__ void encode_slot_lock() {
    while (atomicCAS(&g_encode_slot_lock, 0, 1) != 0) {
    }
}

__device__ __forceinline__ void encode_slot_unlock() {
    __threadfence();
    atomicExch(&g_encode_slot_lock, 0);
}

__device__ __forceinline__ int acquire_encode_slot() {
    encode_slot_lock();
    if (g_encode_slot_top <= 0) {
        encode_slot_unlock();
        return -1;
    }
    --g_encode_slot_top;
    int slot = g_encode_slot_stack[g_encode_slot_top];
    encode_slot_unlock();
    return slot;
}

__device__ __forceinline__ void release_encode_slot(int slot) {
    encode_slot_lock();
    if (g_encode_slot_top >= GTAP_K_ENCODE_SLOTS) {
        encode_slot_unlock();
        mark_overflow();
        return;
    }
    g_encode_slot_stack[g_encode_slot_top] = slot;
    ++g_encode_slot_top;
    encode_slot_unlock();
}

#ifdef GTAP_K_STATS
__device__ __forceinline__ void record_task_profile(
    int kind,
    unsigned long long cycles,
    int u,
    int v,
    int m,
    int a,
    int b,
    int c,
    int d) {
    unsigned long long old = atomicMax(&g_task_max_cycles[kind], cycles);
    if (cycles > old) {
        int base = kind * K_PROFILE_FIELDS;
        g_task_max_info[base + 0] = u;
        g_task_max_info[base + 1] = v;
        g_task_max_info[base + 2] = m;
        g_task_max_info[base + 3] = a;
        g_task_max_info[base + 4] = b;
        g_task_max_info[base + 5] = c;
        g_task_max_info[base + 6] = d;
        g_task_max_info[base + 7] = block_worker_id();
    }
}
#endif
