#pragma once

// Runtime configuration defaults and validation.
// Users may override these macros with compiler -D flags before including GTaP.

#ifndef GTAP_WARP_SIZE
#define GTAP_WARP_SIZE 32
#endif

#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif

#define GTAP_MAX_THREADS_PER_BLOCK 1024
#define GTAP_MAX_WARPS_PER_BLOCK \
    (GTAP_MAX_THREADS_PER_BLOCK / GTAP_WARP_SIZE)

// Internal compatibility bound for experimental runtimes. Public runtimes use
// blockDim.x and d_gtap_launch_config.warps_per_block instead.
#define GTAP_NUM_WARPS GTAP_MAX_WARPS_PER_BLOCK

static_assert(GTAP_WARP_SIZE == 32, "GTAP_WARP_SIZE must be 32 on CUDA");
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

#define GTAP_VALIDATE_THREAD_CONFIG() \
    static_assert(GTAP_NUM_QUEUES > 0, "GTAP_NUM_QUEUES must be positive"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP > 0, "GTAP_MAX_TASKS_PER_WARP must be positive"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP >= GTAP_NUM_QUEUES, "GTAP_MAX_TASKS_PER_WARP must be >= GTAP_NUM_QUEUES"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP % GTAP_NUM_QUEUES == 0, "GTAP_MAX_TASKS_PER_WARP must be divisible by GTAP_NUM_QUEUES")

#define GTAP_VALIDATE_BLOCK_CONFIG() \
    static_assert(GTAP_MAX_TASKS_PER_BLOCK > 0, "GTAP_MAX_TASKS_PER_BLOCK must be positive")
