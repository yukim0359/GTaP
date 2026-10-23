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

#ifndef GTAP_ADJACENCY_SLOTS
#define GTAP_ADJACENCY_SLOTS 4096
#endif

#ifndef GTAP_MAX_PREFIX_DEPTH
#define GTAP_MAX_PREFIX_DEPTH 24
#endif

#ifndef GTAP_CUDA_STACK_SIZE
#define GTAP_CUDA_STACK_SIZE 4096
#endif

#ifndef GTAP_NCR_ROWS
#define GTAP_NCR_ROWS 1001
#endif

#ifndef GTAP_NCR_COLS
#define GTAP_NCR_COLS 401
#endif
