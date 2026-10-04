#pragma once

#ifndef __GTAP_IS_THREAD_MODE
#define __GTAP_IS_THREAD_MODE
#endif

#define GTAP_DETAIL_THREAD_BACKEND_EXPERIMENTAL_GLOBAL_QUEUE

#include "../detail/thread/scheduler.cuh"
#include "../detail/thread/lifecycle.cuh"
#include "../detail/thread/host_api.cuh"
#include "../detail/thread/compiler_api.cuh"
#include "../detail/thread/profile_export.cuh"
