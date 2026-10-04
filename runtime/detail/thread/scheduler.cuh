#pragma once

// The public header sets one of these. With neither set, use the default scheduler.
#if defined(GTAP_DETAIL_THREAD_BACKEND_EXPERIMENTAL_GLOBAL_QUEUE) && defined(GTAP_DETAIL_THREAD_BACKEND_EXPERIMENTAL_CHASE_LEV)
#error "Select one thread backend"
#elif defined(GTAP_DETAIL_THREAD_BACKEND_EXPERIMENTAL_GLOBAL_QUEUE)
#include "backends/experimental/global_queue/scheduler.cuh"
#elif defined(GTAP_DETAIL_THREAD_BACKEND_EXPERIMENTAL_CHASE_LEV)
#include "backends/experimental/chase_lev/scheduler.cuh"
#else
#include "backends/default/scheduler.cuh"
#endif
