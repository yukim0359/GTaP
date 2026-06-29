#pragma once

// Runtime configuration defaults and validation.
// Users may override these macros with compiler -D flags before including GTaP.

#ifndef GTAP_WARP_SIZE
#define GTAP_WARP_SIZE 32
#endif

#ifndef GTAP_GRID_SIZE
#define GTAP_GRID_SIZE 1024
#endif

#ifndef GTAP_BLOCK_SIZE
#define GTAP_BLOCK_SIZE 256
#endif

#ifndef GTAP_MAX_CHILD_TASKS
#define GTAP_MAX_CHILD_TASKS 32
#endif

#ifdef PROFILE
#ifndef MAX_PROFILE_DATA
#define MAX_PROFILE_DATA 30000
#endif
#endif

#define GTAP_NUM_WARPS ((GTAP_BLOCK_SIZE + GTAP_WARP_SIZE - 1) / GTAP_WARP_SIZE)

static_assert(GTAP_WARP_SIZE == 32, "GTAP_WARP_SIZE must be 32 on CUDA");
static_assert(GTAP_GRID_SIZE > 0, "GTAP_GRID_SIZE must be positive");
static_assert(GTAP_BLOCK_SIZE > 0, "GTAP_BLOCK_SIZE must be positive");
static_assert(GTAP_BLOCK_SIZE % GTAP_WARP_SIZE == 0, "GTAP_BLOCK_SIZE must be a multiple of GTAP_WARP_SIZE");
static_assert(GTAP_NUM_WARPS > 0, "GTAP_NUM_WARPS must be positive");
static_assert(GTAP_MAX_CHILD_TASKS >= 0, "GTAP_MAX_CHILD_TASKS must be non-negative");

#ifdef PROFILE
static_assert(MAX_PROFILE_DATA > 0, "MAX_PROFILE_DATA must be positive when PROFILE is enabled");
#endif

#define GTAP_VALIDATE_RESULT_HANDLE_CONFIG() \
    static_assert(GTAP_RESULT_HANDLE_CAPACITY > 0, "GTAP_RESULT_HANDLE_CAPACITY must be positive")

#define GTAP_VALIDATE_THREAD_CONFIG() \
    static_assert(GTAP_NUM_QUEUES > 0, "GTAP_NUM_QUEUES must be positive"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP > 0, "GTAP_MAX_TASKS_PER_WARP must be positive"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP >= GTAP_NUM_QUEUES, "GTAP_MAX_TASKS_PER_WARP must be >= GTAP_NUM_QUEUES"); \
    static_assert(GTAP_MAX_TASKS_PER_WARP % GTAP_NUM_QUEUES == 0, "GTAP_MAX_TASKS_PER_WARP must be divisible by GTAP_NUM_QUEUES")

#define GTAP_VALIDATE_BLOCK_CONFIG() \
    static_assert(GTAP_MAX_TASKS_PER_BLOCK > 0, "GTAP_MAX_TASKS_PER_BLOCK must be positive")
