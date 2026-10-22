#pragma once

// Smoke-run compile-time defaults. Makefile -D overrides these.
// Paper graph×k values live in data/gtap_graph_k_config.csv and are not
// applied unless a script rebuilds with those -D flags.
#ifndef GTAP_K_RANGE_CUTOFF
#define GTAP_K_RANGE_CUTOFF 128
#endif

#ifndef GTAP_K_TASK_FANOUT
#define GTAP_K_TASK_FANOUT 128
#endif

#ifndef GTAP_K_MAX_CANDIDATES
#define GTAP_K_MAX_CANDIDATES 512
#endif

#ifndef GTAP_K_BIT_BUFFER_COUNT
#define GTAP_K_BIT_BUFFER_COUNT 256
#endif

#ifndef GTAP_K_ENCODE_SLOTS
#define GTAP_K_ENCODE_SLOTS 4096
#endif

#ifndef GTAP_K_RANGE_MAX_FIXED
#define GTAP_K_RANGE_MAX_FIXED 24
#endif

#ifndef GTAP_K_CUDA_STACK_SIZE
#define GTAP_K_CUDA_STACK_SIZE 4096
#endif

// Thread launch config (Makefile sets these per binary via -D).
#ifndef GTAP_K_RUNTIME_GRID_SIZE
#define GTAP_K_RUNTIME_GRID_SIZE 2640
#endif

#ifndef GTAP_K_RUNTIME_BLOCK_SIZE
#define GTAP_K_RUNTIME_BLOCK_SIZE 32
#endif

#ifndef GTAP_K_RUNTIME_MAX_TASKS_PER_WARP
#define GTAP_K_RUNTIME_MAX_TASKS_PER_WARP 150000
#endif

#ifndef GTAP_K_RUNTIME_NUM_QUEUES
#define GTAP_K_RUNTIME_NUM_QUEUES 1
#endif

// Orientation-only defaults.
#ifndef GTAP_K_WAIT_FANOUT
#define GTAP_K_WAIT_FANOUT 32
#endif

#ifndef GTAP_K_HEAVY_CANDIDATES
#define GTAP_K_HEAVY_CANDIDATES 32
#endif

#ifndef GTAP_K_SECOND_HEAVY_CANDIDATES
#define GTAP_K_SECOND_HEAVY_CANDIDATES 32
#endif

#ifndef GTAP_K_THIRD_HEAVY_CANDIDATES
#define GTAP_K_THIRD_HEAVY_CANDIDATES 32
#endif

// Pivot-only defaults.
#ifndef GTAP_K_PIVOT_PREFIX_CHUNK
#define GTAP_K_PIVOT_PREFIX_CHUNK 16
#endif

#ifndef GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES
#define GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES 64
#endif

#ifndef GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES
#define GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES 8
#endif

#ifndef GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD
#define GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD 1024
#endif

#ifndef GTAP_K_PIVOT_PREFIX_FANOUT
#define GTAP_K_PIVOT_PREFIX_FANOUT 128
#endif

#ifndef GTAP_K_PIVOT_STATE_LEVELS
#define GTAP_K_PIVOT_STATE_LEVELS 128
#endif

// Pivot nCr lookup table shape (host load + device lookup must match).
#ifndef GTAP_K_NCR_ROWS
#define GTAP_K_NCR_ROWS 1001
#endif

#ifndef GTAP_K_NCR_COLS
#define GTAP_K_NCR_COLS 401
#endif
