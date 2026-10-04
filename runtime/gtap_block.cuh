#pragma once

#ifndef __GTAP_IS_BLOCK_MODE
#define __GTAP_IS_BLOCK_MODE
#endif

#ifdef GTAP_DETAIL_BLOCK_BACKEND_EXPERIMENTAL_GLOBAL_QUEUE
#error "A different block backend is already selected"
#endif

#define GTAP_DETAIL_BLOCK_BACKEND_DEFAULT

#include "detail/block/scheduler.cuh"
#include "detail/block/lifecycle.cuh"
#include "detail/block/host_api.cuh"
#include "detail/block/compiler_api.cuh"
#include "detail/block/profile_export.cuh"
