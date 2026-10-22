# GTaP k-clique benchmarks. See README.md for variants, data, and timing.
#
# Two layers of knobs:
#   1. Build defaults in this file / common/config_defaults.cuh
#      and thread/config_defaults.cuh
#   2. Paper graph×k overrides in data/gtap_graph_k_config.csv
#      (applied by scripts/run_benchmark.sh via load_gtap_graph_k_config.py).
#
# Runtime (no rebuild): GTAP_ORIENT, GTAP_GRID_SIZE, GTAP_BLOCK_SIZE,
# GTAP_MAX_TASKS_PER_WARP. Each binary has its own launch default.
# The binaries are orientation vs pivot (see README.md).

BENCHMARK_NAME := k_clique

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

OUT_DIR    := $(SCRIPT_DIR)/bin

KCGPU_ROOT ?= $(abspath $(SCRIPT_DIR)/../../../../KCGPU)
NVCC       ?= nvcc
NVCC_ARCH_CODE := $(patsubst sm_%,%,$(CUDA_ARCH))

COMMON_DIR := $(SCRIPT_DIR)/common
THREAD_DIR := $(SCRIPT_DIR)/thread

SRC_GPU_ORIENT := $(COMMON_DIR)/gpu_orient.cu
SRC_KCGPU_ORIENT := $(COMMON_DIR)/kcgpu_orient/kcgpu_orient.cu
OBJ_KCGPU_ORIENT := $(OUT_DIR)/kcgpu_orient.o
LIB_KCGPU_ORIENT := $(OUT_DIR)/libkcgpu_orient.a
SRC_GTAP_ORIENTATION := $(THREAD_DIR)/orientation/main.cu
SRC_GTAP_PIVOT := $(THREAD_DIR)/pivot/main.cu

OUT_GTAP_ORIENTATION := $(OUT_DIR)/gtap_orientation
OUT_GTAP_PIVOT := $(OUT_DIR)/gtap_pivot

# --- common GTaP k-clique tuning ---
#
# Task split at the edge-list level (edgek_range); both pivot and orientation
# use one edgek_edge per edge.
# |edge range| > this -> split into edgek_range subtasks.
GTAP_RANGE_CUTOFF ?= 128
# Max subtasks when chunking ranges / encoded-first (no slot).
GTAP_TASK_FANOUT ?= 128
#
# Per-edge (u,v): cap on common-neighbor count m.
GTAP_MAX_CANDIDATES ?= 512
#
# Encoded graph scratch (enc size follows GTAP_MAX_CANDIDATES):
# Pivot levels / encoded recursion levels (buf_level).
GTAP_BIT_BUFFER_COUNT ?= 256
# Shared encode buffers; slot failure -> per-worker encode_buffer path.
GTAP_ENCODE_SLOTS ?= 4096

# --- orientation-only split thresholds ---
# m >= this -> encode+slot path (else inline encoded count).
GTAP_HEAVY_CANDIDATES ?= 32
# Chunk width for shared_encoded first/second/third (slot path).
GTAP_WAIT_FANOUT ?= 32
# 1st-level child_count >= this -> shared_encoded_second_range.
GTAP_SECOND_HEAVY_CANDIDATES ?= 32
# 2nd-level second_count >= this -> shared_encoded_third_range.
GTAP_THIRD_HEAVY_CANDIDATES ?= 32

# --- pivot-only split thresholds ---
# Lower values create more tasks inside heavy per-edge pivot searches,
# reducing tail work at the cost of scheduler overhead.
# Candidate indices per initial pivot_shared_prefix subtask (chunks [0,m) at depth 1).
GTAP_PIVOT_PREFIX_CHUNK ?= 16
# m >= this -> encode slot + pivot_shared_prefix (else inline pivot count).
GTAP_PIVOT_SHARED_SPLIT_CANDIDATES ?= 64
# child cl/pl/intersection count >= this -> prefix branch may spawn subtasks.
GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES ?= 8
# Min work_estimate to spawn pivot_shared_prefix during prefix DFS.
GTAP_PIVOT_PREFIX_WORK_THRESHOLD ?= 1024
# Max subtasks when splitting a child PL range (chunk = ceil(pl_count / fanout)).
GTAP_PIVOT_PREFIX_FANOUT ?= 128
# Max prefix depth stored in KRangeFixed.idx[]; also caps pivot prefix spawn depth.
GTAP_RANGE_MAX_FIXED ?= 24
# Pivot prefix state levels (cl/pl/pivot scratch per level; <= BIT_BUFFER_COUNT/2).
GTAP_PIVOT_STATE_LEVELS ?= 128
#
# Runtime limits:
GTAP_ORIENTATION_CUDA_STACK_SIZE ?= 4096
GTAP_PIVOT_CUDA_STACK_SIZE       ?= 1024
GTAP_CFLAGS                      ?=
GTAP_EXTRA_CFLAGS                ?=
# Set to 1 to enable candidate/task stats in both GTaP variants.
GTAP_STATS            ?= 0
ifeq ($(GTAP_STATS),1)
GTAP_EXTRA_CFLAGS += -DGTAP_STATS
endif
# Set to 1 to enable GTaP warp timeline profiling (benchmarks should keep this 0).
GTAP_ENABLE_PROFILING            ?= 0
ifeq ($(GTAP_ENABLE_PROFILING),1)
GTAP_EXTRA_CFLAGS += -DGTAP_ENABLE_PROFILING
else
GTAP_EXTRA_CFLAGS += -UGTAP_ENABLE_PROFILING
endif

NCR_TXT := $(SCRIPT_DIR)/data/nCr.txt
SRC_GEN_NCR := $(SCRIPT_DIR)/scripts/gen_ncr.py
GTAP_NCR_ROWS ?= 1001
GTAP_NCR_COLS ?= 401

.PHONY: all gtap_orientation gtap_pivot ncr clean print-vars

all: gtap_orientation gtap_pivot

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CXX_BIN=$(CXX_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"
	@echo "GTAP_ENABLE_PROFILING=$(GTAP_ENABLE_PROFILING)"
	@echo "GTAP_MAX_CANDIDATES=$(GTAP_MAX_CANDIDATES)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OBJ_KCGPU_ORIENT): $(SRC_KCGPU_ORIENT) | $(OUT_DIR)
	"$(NVCC)" -std=c++17 -O3 -w -Xcompiler -fPIC \
	  -I"$(KCGPU_ROOT)/include" -I"$(KCGPU_ROOT)/kcore" \
	  -arch=sm_$(NVCC_ARCH_CODE) \
	  -DCUDA -DNOT_IMPL \
	  -c "$(SRC_KCGPU_ORIENT)" -o "$(OBJ_KCGPU_ORIENT)"

$(LIB_KCGPU_ORIENT): $(OBJ_KCGPU_ORIENT)
	ar rcs "$(LIB_KCGPU_ORIENT)" "$(OBJ_KCGPU_ORIENT)"

gtap_orientation: $(OUT_GTAP_ORIENTATION)

$(OUT_GTAP_ORIENTATION): $(SRC_GTAP_ORIENTATION) $(SRC_GPU_ORIENT) $(LIB_KCGPU_ORIENT) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -I"$(COMMON_DIR)" \
	  -I"$(THREAD_DIR)" \
	  -DGTAP_RANGE_CUTOFF="$(GTAP_RANGE_CUTOFF)" \
	  -DGTAP_TASK_FANOUT="$(GTAP_TASK_FANOUT)" \
	  -DGTAP_WAIT_FANOUT="$(GTAP_WAIT_FANOUT)" \
	  -DGTAP_MAX_CANDIDATES="$(GTAP_MAX_CANDIDATES)" \
	  -DGTAP_BIT_BUFFER_COUNT="$(GTAP_BIT_BUFFER_COUNT)" \
	  -DGTAP_HEAVY_CANDIDATES="$(GTAP_HEAVY_CANDIDATES)" \
	  -DGTAP_SECOND_HEAVY_CANDIDATES="$(GTAP_SECOND_HEAVY_CANDIDATES)" \
	  -DGTAP_THIRD_HEAVY_CANDIDATES="$(GTAP_THIRD_HEAVY_CANDIDATES)" \
	  -DGTAP_ENCODE_SLOTS="$(GTAP_ENCODE_SLOTS)" \
	  -DGTAP_RANGE_MAX_FIXED="$(GTAP_RANGE_MAX_FIXED)" \
	  -DGTAP_CUDA_STACK_SIZE="$(GTAP_ORIENTATION_CUDA_STACK_SIZE)" \
	  $(GTAP_CFLAGS) $(GTAP_EXTRA_CFLAGS) \
	  "$(SRC_GTAP_ORIENTATION)" "$(SRC_GPU_ORIENT)" -o "$(OUT_GTAP_ORIENTATION)" \
	  -L"$(CUDA_PATH)/lib64" -L"$(OUT_DIR)" -lcudart -lkcgpu_orient

gtap_pivot: $(OUT_GTAP_PIVOT)

$(OUT_GTAP_PIVOT): $(SRC_GTAP_PIVOT) $(SRC_GPU_ORIENT) $(LIB_KCGPU_ORIENT) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -I"$(COMMON_DIR)" \
	  -I"$(THREAD_DIR)" \
	  -DGTAP_RANGE_CUTOFF="$(GTAP_RANGE_CUTOFF)" \
	  -DGTAP_TASK_FANOUT="$(GTAP_TASK_FANOUT)" \
	  -DGTAP_MAX_CANDIDATES="$(GTAP_MAX_CANDIDATES)" \
	  -DGTAP_BIT_BUFFER_COUNT="$(GTAP_BIT_BUFFER_COUNT)" \
	  -DGTAP_ENCODE_SLOTS="$(GTAP_ENCODE_SLOTS)" \
	  -DGTAP_PIVOT_PREFIX_CHUNK="$(GTAP_PIVOT_PREFIX_CHUNK)" \
	  -DGTAP_PIVOT_SHARED_SPLIT_CANDIDATES="$(GTAP_PIVOT_SHARED_SPLIT_CANDIDATES)" \
	  -DGTAP_PIVOT_PREFIX_HEAVY_CANDIDATES="$(GTAP_PIVOT_PREFIX_HEAVY_CANDIDATES)" \
	  -DGTAP_PIVOT_PREFIX_WORK_THRESHOLD="$(GTAP_PIVOT_PREFIX_WORK_THRESHOLD)" \
	  -DGTAP_PIVOT_PREFIX_FANOUT="$(GTAP_PIVOT_PREFIX_FANOUT)" \
	  -DGTAP_RANGE_MAX_FIXED="$(GTAP_RANGE_MAX_FIXED)" \
	  -DGTAP_PIVOT_STATE_LEVELS="$(GTAP_PIVOT_STATE_LEVELS)" \
	  -DGTAP_CUDA_STACK_SIZE="$(GTAP_PIVOT_CUDA_STACK_SIZE)" \
	  $(GTAP_CFLAGS) $(GTAP_EXTRA_CFLAGS) \
	  "$(SRC_GTAP_PIVOT)" "$(SRC_GPU_ORIENT)" -o "$(OUT_GTAP_PIVOT)" \
	  -L"$(CUDA_PATH)/lib64" -L"$(OUT_DIR)" -lcudart -lkcgpu_orient

ncr: $(NCR_TXT)

$(NCR_TXT): $(SRC_GEN_NCR)
	mkdir -p "$(SCRIPT_DIR)/data"
	python3 "$(SRC_GEN_NCR)" --rows $(GTAP_NCR_ROWS) --cols $(GTAP_NCR_COLS) "$(NCR_TXT)"

clean:
	rm -f $(OUT_GTAP_ORIENTATION) $(OUT_GTAP_PIVOT) $(OBJ_KCGPU_ORIENT) $(LIB_KCGPU_ORIENT)
