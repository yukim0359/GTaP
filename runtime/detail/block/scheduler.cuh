#pragma once

// The public header sets this. With it unset, use the default scheduler.
#ifdef GTAP_DETAIL_BLOCK_BACKEND_EXPERIMENTAL_GLOBAL_QUEUE
#include "backends/experimental/global_queue/scheduler.cuh"
#else
#include "backends/default/scheduler.cuh"
#endif
