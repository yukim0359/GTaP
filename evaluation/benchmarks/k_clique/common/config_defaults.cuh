#pragma once

// Shared smoke-run compile-time defaults. Makefile -D overrides these.
// Mode-specific values live in thread/config_defaults.cuh and
// block/config_defaults.cuh. Paper graph×k values live in
// data/gtap_graph_k_config.csv and are applied only when a script rebuilds
// with those -D flags.

#ifndef GTAP_RANGE_CUTOFF
#define GTAP_RANGE_CUTOFF 128
#endif

#ifndef GTAP_TASK_FANOUT
#define GTAP_TASK_FANOUT 128
#endif

#ifndef GTAP_MAX_CANDIDATES
#define GTAP_MAX_CANDIDATES 512
#endif

#ifndef GTAP_BIT_BUFFER_COUNT
#define GTAP_BIT_BUFFER_COUNT 256
#endif

#ifndef GTAP_ENCODE_SLOTS
#define GTAP_ENCODE_SLOTS 4096
#endif

#ifndef GTAP_RANGE_MAX_FIXED
#define GTAP_RANGE_MAX_FIXED 24
#endif

#ifndef GTAP_CUDA_STACK_SIZE
#define GTAP_CUDA_STACK_SIZE 4096
#endif

#ifndef GTAP_PIVOT_STATE_LEVELS
#define GTAP_PIVOT_STATE_LEVELS 128
#endif

#ifndef GTAP_NCR_ROWS
#define GTAP_NCR_ROWS 1001
#endif

#ifndef GTAP_NCR_COLS
#define GTAP_NCR_COLS 401
#endif
