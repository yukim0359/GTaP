#pragma once

#include "fmm_real.cuh"

#ifndef FMM3D_SERIAL_HOST_DTT
#define FMM3D_SERIAL_HOST_DTT 0
#endif

#ifndef FMM3D_WORKLIST_DTT
#define FMM3D_WORKLIST_DTT 0
#endif

// ---------- Tunable parameters ----------

#ifndef FMM3D_ENABLE_GTAP_DTT
#define FMM3D_ENABLE_GTAP_DTT 1
#endif
#ifndef FMM3D_DTT_TASK_DEPTH
#define FMM3D_DTT_TASK_DEPTH 5
#endif
#ifndef FMM3D_DTT_TASK_MIN_N
#define FMM3D_DTT_TASK_MIN_N 4096
#endif
#ifndef FMM3D_HOST_DTT_TASK_DEPTH
#define FMM3D_HOST_DTT_TASK_DEPTH 5
#endif
#ifndef FMM3D_HOST_DTT_TASK_MIN_N
#define FMM3D_HOST_DTT_TASK_MIN_N 4096
#endif
#ifndef FMM3D_DTT_M2L_CAP
#define FMM3D_DTT_M2L_CAP 1000
#endif
#ifndef FMM3D_DTT_P2P_CAP
#define FMM3D_DTT_P2P_CAP 500
#endif
#ifndef FMM3D_NCRIT
#define FMM3D_NCRIT 32
#endif
#ifndef FMM3D_SET_CUDA_STACK_LIMIT
#define FMM3D_SET_CUDA_STACK_LIMIT 0
#endif
#ifndef FMM3D_CUDA_STACK_SIZE
#define FMM3D_CUDA_STACK_SIZE 8192
#endif
#ifndef FMM3D_P
#define FMM3D_P 3
#endif
#ifndef FMM3D_FUSED_DTT
#define FMM3D_FUSED_DTT 0
#endif
#ifndef FMM3D_PARTITIONED_FUSED_DTT
#define FMM3D_PARTITIONED_FUSED_DTT 0
#endif
#if FMM3D_PARTITIONED_FUSED_DTT && !FMM3D_FUSED_DTT
#error "FMM3D_PARTITIONED_FUSED_DTT requires FMM3D_FUSED_DTT=1"
#endif
#ifndef FMM3D_DTT_NSPAWN
#define FMM3D_DTT_NSPAWN 10000
#endif
#ifndef FMM3D_WORKLIST_PAIR_CAP_FACTOR
#define FMM3D_WORKLIST_PAIR_CAP_FACTOR 64
#endif
#ifndef FMM3D_M2L_BLOCK_SIZE
#define FMM3D_M2L_BLOCK_SIZE 64
#endif
#ifndef FMM3D_GPU_L2L
#define FMM3D_GPU_L2L 1
#endif
#ifndef FMM3D_GPU_UPWARD
#define FMM3D_GPU_UPWARD 1
#endif
#ifndef FMM3D_GPU_TREE_BUILD
#define FMM3D_GPU_TREE_BUILD 1
#endif
#ifndef FMM3D_M2L_COMPACT_TARGETS
#define FMM3D_M2L_COMPACT_TARGETS 0
#endif
#ifndef FMM3D_MORTON_BITS
#define FMM3D_MORTON_BITS 21
#endif
#ifndef FMM3D_DIRECT_VALIDATE_N
#define FMM3D_DIRECT_VALIDATE_N 4096
#endif
#ifndef FMM3D_DIRECT_SAMPLE_N
#define FMM3D_DIRECT_SAMPLE_N 0
#endif
#ifndef FMM3D_DIRECT_SAMPLE_SEED
#define FMM3D_DIRECT_SAMPLE_SEED 12345
#endif
#ifndef FMM3D_DIRECT_SAMPLE_BLOCK_SIZE
#define FMM3D_DIRECT_SAMPLE_BLOCK_SIZE 256
#endif
#ifndef FMM3D_BODY_SEED
#define FMM3D_BODY_SEED 42
#endif
#ifndef FMM3D_BODY_SEED_PER_N
#define FMM3D_BODY_SEED_PER_N 1
#endif
#ifndef GTAP_BENCH_GRID_SIZE
#define GTAP_BENCH_GRID_SIZE 3168
#endif
#ifndef GTAP_BENCH_BLOCK_SIZE
#define GTAP_BENCH_BLOCK_SIZE 32
#endif
#ifndef GTAP_BENCH_MAX_TASKS_PER_WARP
#define GTAP_BENCH_MAX_TASKS_PER_WARP 50000
#endif
#ifndef GTAP_BENCH_NUM_QUEUES
#define GTAP_BENCH_NUM_QUEUES 1
#endif
#ifndef GTAP_BENCH_PROFILE_CAPACITY_PER_WARP
#define GTAP_BENCH_PROFILE_CAPACITY_PER_WARP 30000
#endif
#if FMM3D_P < 1
#error "FMM3D_P must be at least 1"
#endif
#if FMM3D_DIRECT_SAMPLE_BLOCK_SIZE < 32 || \
    (FMM3D_DIRECT_SAMPLE_BLOCK_SIZE & (FMM3D_DIRECT_SAMPLE_BLOCK_SIZE - 1)) != 0
#error "FMM3D_DIRECT_SAMPLE_BLOCK_SIZE must be a power of two and at least 32"
#endif
#define FMM3D_NTERM (FMM3D_P * (FMM3D_P + 1) / 2)

static constexpr int NCRIT = FMM3D_NCRIT;
static constexpr int MAX_DEPTH = 24;
static constexpr int EXA_P = FMM3D_P;
static constexpr int EXA_NTERM = FMM3D_NTERM;
static constexpr int EXA_NSPH = 4 * EXA_P * EXA_P;
static constexpr int M2L_CNM_SIZE = EXA_P * EXA_P * EXA_P * EXA_P;
static constexpr int FMM_CONST_BYTES =
    M2L_CNM_SIZE * 2 * static_cast<int>(sizeof(Real)) +
    EXA_NSPH * static_cast<int>(sizeof(Real));
static constexpr bool FMM_CONST_IN_CONSTANT = FMM_CONST_BYTES <= 65536;
static constexpr int DTT_TASK_DEPTH = FMM3D_DTT_TASK_DEPTH;
static constexpr int DTT_TASK_MIN_N = FMM3D_DTT_TASK_MIN_N;
static constexpr int HOST_DTT_TASK_DEPTH = FMM3D_HOST_DTT_TASK_DEPTH;
static constexpr int HOST_DTT_TASK_MIN_N = FMM3D_HOST_DTT_TASK_MIN_N;
static constexpr int DTT_M2L_CAP = FMM3D_DTT_M2L_CAP;
static constexpr int DTT_P2P_CAP = FMM3D_DTT_P2P_CAP;
[[maybe_unused]] static constexpr int WORKLIST_PAIR_CAP_FACTOR = FMM3D_WORKLIST_PAIR_CAP_FACTOR;
static constexpr int M2L_BLOCK_SIZE = FMM3D_M2L_BLOCK_SIZE;
static constexpr int MORTON_BITS = FMM3D_MORTON_BITS;
static constexpr int MORTON_BITS_MAX = 21;  // uint64 Morton key uses 3 bits per level
static_assert(FMM3D_MORTON_BITS >= 1 && FMM3D_MORTON_BITS <= MORTON_BITS_MAX,
              "FMM3D_MORTON_BITS must be in [1, 21] (64-bit Morton key limit)");
static constexpr int DIRECT_VALIDATE_N = FMM3D_DIRECT_VALIDATE_N;
static constexpr int DIRECT_SAMPLE_SEED = FMM3D_DIRECT_SAMPLE_SEED;
static constexpr int DIRECT_SAMPLE_BLOCK_SIZE = FMM3D_DIRECT_SAMPLE_BLOCK_SIZE;
static constexpr int DTT_NSPAWN = FMM3D_DTT_NSPAWN;

#define FMM3D_GTAP_FUSED (FMM3D_ENABLE_GTAP_DTT && FMM3D_FUSED_DTT)
#define FMM3D_GTAP_ATOMIC_FUSED (FMM3D_GTAP_FUSED && !FMM3D_PARTITIONED_FUSED_DTT)
