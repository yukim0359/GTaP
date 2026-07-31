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

SRC_GPU_ORIENT := $(SCRIPT_DIR)/k_clique_gpu_orient.cu
SRC_KCGPU_ORIENT := $(SCRIPT_DIR)/kcgpu_orient/kcgpu_orient.cu
OBJ_KCGPU_ORIENT := $(OUT_DIR)/kcgpu_orient.o
LIB_KCGPU_ORIENT := $(OUT_DIR)/libkcgpu_orient.a
SRC_GTAP_ORIENTATION := $(SCRIPT_DIR)/gtap_k_clique_orientation.cu
SRC_GTAP_PIVOT := $(SCRIPT_DIR)/gtap_k_clique_pivot.cu

OUT_GTAP_ORIENTATION := $(OUT_DIR)/gtap_orientation
OUT_GTAP_PIVOT := $(OUT_DIR)/gtap_pivot

# --- GTaP runtime launch / queue sizing (per binary) ---
# Passed to gtap_thread_config through benchmark-local compile-time defaults.
# blocks for gtap_orientation
GTAP_ORIENTATION_GRID_SIZE          ?= 2640
GTAP_ORIENTATION_BLOCK_SIZE         ?= 32
GTAP_ORIENTATION_MAX_TASKS_PER_WARP ?= 150000
# blocks for gtap_pivot
GTAP_PIVOT_GRID_SIZE                ?= 1600
GTAP_PIVOT_BLOCK_SIZE               ?= 32
GTAP_PIVOT_MAX_TASKS_PER_WARP       ?= 200000
# worker queues (shared by both binaries)
GTAP_NUM_QUEUES                     ?= 1

# --- common GTaP k-clique tuning ---
#
# Task split at the edge-list level (edgek_range); both pivot and orientation
# use one edgek_edge per edge.
# |edge range| > this -> split into edgek_range subtasks.
GTAP_K_RANGE_CUTOFF ?= 128
# Max subtasks when chunking ranges / encoded-first (no slot).
GTAP_K_TASK_FANOUT ?= 128
#
# Per-edge (u,v): cap on common-neighbor count m.
GTAP_K_MAX_CANDIDATES ?= 512
#
# Encoded graph scratch (enc size follows GTAP_K_MAX_CANDIDATES):
# Pivot levels / encoded recursion levels (buf_level).
GTAP_K_BIT_BUFFER_COUNT ?= 256
# Shared encode buffers; slot failure -> per-worker encode_buffer path.
GTAP_K_ENCODE_SLOTS ?= 4096

# --- orientation-only split thresholds ---
# m >= this -> encode+slot path (else inline encoded count).
GTAP_K_HEAVY_CANDIDATES ?= 32
# Chunk width for shared_encoded first/second/third (slot path).
GTAP_K_WAIT_FANOUT ?= 32
# 1st-level child_count >= this -> shared_encoded_second_range.
GTAP_K_SECOND_HEAVY_CANDIDATES ?= 32
# 2nd-level second_count >= this -> shared_encoded_third_range.
GTAP_K_THIRD_HEAVY_CANDIDATES ?= 32

# --- pivot-only split thresholds ---
# Lower values create more tasks inside heavy per-edge pivot searches,
# reducing tail work at the cost of scheduler overhead.
# Candidate indices per initial pivot_shared_prefix subtask (chunks [0,m) at depth 1).
GTAP_K_PIVOT_PREFIX_CHUNK ?= 16
# m >= this -> encode slot + pivot_shared_prefix (else inline pivot count).
GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES ?= 64
# child cl/pl/intersection count >= this -> prefix branch may spawn subtasks.
GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES ?= 8
# Min work_estimate to spawn pivot_shared_prefix during prefix DFS.
GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD ?= 1024
# Max subtasks when splitting a child PL range (chunk = ceil(pl_count / fanout)).
GTAP_K_PIVOT_PREFIX_FANOUT ?= 128
# 1: pass pl/cl prefix state by value to prefix tasks.
# 0: old path; pass KRangeFixed branch indices and rebuild the prefix state in each task.
GTAP_K_PIVOT_PREFIX_PASS_STATE ?= 0
# Max prefix depth stored in KRangeFixed.idx[]; also caps pivot prefix spawn depth.
GTAP_K_RANGE_MAX_FIXED ?= 24
# Pivot prefix state levels (cl/pl/pivot scratch per level; <= BIT_BUFFER_COUNT/2).
GTAP_K_PIVOT_STATE_LEVELS ?= 128
#
# Runtime limits:
GTAP_ORIENTATION_CUDA_STACK_SIZE ?= 4096
GTAP_PIVOT_CUDA_STACK_SIZE       ?= 1024
GTAP_CFLAGS                      ?=
GTAP_EXTRA_CFLAGS                ?=
GTAP_ORIENT                      ?= degen
ifeq ($(GTAP_ORIENT),degree)
GTAP_DEFAULT_ORIENT := GTAP_ORIENT_DEGREE
else ifeq ($(GTAP_ORIENT),degen)
GTAP_DEFAULT_ORIENT := GTAP_ORIENT_DEGEN
else
$(error GTAP_ORIENT must be degree or degen, got "$(GTAP_ORIENT)")
endif
# Set to 1 to enable recurse/candidate/task stats in both GTaP variants.
GTAP_K_STATS            ?= 0
ifeq ($(GTAP_K_STATS),1)
GTAP_EXTRA_CFLAGS += -DGTAP_K_STATS
endif
# Set to 1 to enable GTaP warp timeline profiling (benchmarks should keep this 0).
GTAP_PROFILE            ?= 0
ifeq ($(GTAP_PROFILE),1)
GTAP_EXTRA_CFLAGS += -DPROFILE
else
GTAP_EXTRA_CFLAGS += -UPROFILE
endif

.PHONY: all gtap_orientation gtap_pivot clean print-vars

all: gtap_orientation gtap_pivot

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CXX_BIN=$(CXX_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"
	@echo "GTAP_ORIENT=$(GTAP_ORIENT)"
	@echo "GTAP_PROFILE=$(GTAP_PROFILE)"
	@echo "GTAP_ORIENTATION_GRID_SIZE=$(GTAP_ORIENTATION_GRID_SIZE)"
	@echo "GTAP_ORIENTATION_BLOCK_SIZE=$(GTAP_ORIENTATION_BLOCK_SIZE)"
	@echo "GTAP_ORIENTATION_MAX_TASKS_PER_WARP=$(GTAP_ORIENTATION_MAX_TASKS_PER_WARP)"
	@echo "GTAP_PIVOT_GRID_SIZE=$(GTAP_PIVOT_GRID_SIZE)"
	@echo "GTAP_PIVOT_BLOCK_SIZE=$(GTAP_PIVOT_BLOCK_SIZE)"
	@echo "GTAP_PIVOT_MAX_TASKS_PER_WARP=$(GTAP_PIVOT_MAX_TASKS_PER_WARP)"
	@echo "GTAP_K_MAX_CANDIDATES=$(GTAP_K_MAX_CANDIDATES)"
	@echo "GTAP_K_PIVOT_PREFIX_PASS_STATE=$(GTAP_K_PIVOT_PREFIX_PASS_STATE)"

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
	  -I"$(SCRIPT_DIR)" \
	  -DGTAP_K_RUNTIME_GRID_SIZE="$(GTAP_ORIENTATION_GRID_SIZE)" \
	  -DGTAP_K_RUNTIME_BLOCK_SIZE="$(GTAP_ORIENTATION_BLOCK_SIZE)" \
	  -DGTAP_K_RUNTIME_MAX_TASKS_PER_WARP="$(GTAP_ORIENTATION_MAX_TASKS_PER_WARP)" \
	  -DGTAP_K_RUNTIME_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  -DGTAP_K_RANGE_CUTOFF="$(GTAP_K_RANGE_CUTOFF)" \
	  -DGTAP_K_TASK_FANOUT="$(GTAP_K_TASK_FANOUT)" \
	  -DGTAP_K_WAIT_FANOUT="$(GTAP_K_WAIT_FANOUT)" \
	  -DGTAP_K_MAX_CANDIDATES="$(GTAP_K_MAX_CANDIDATES)" \
	  -DGTAP_K_BIT_BUFFER_COUNT="$(GTAP_K_BIT_BUFFER_COUNT)" \
	  -DGTAP_K_HEAVY_CANDIDATES="$(GTAP_K_HEAVY_CANDIDATES)" \
	  -DGTAP_K_SECOND_HEAVY_CANDIDATES="$(GTAP_K_SECOND_HEAVY_CANDIDATES)" \
	  -DGTAP_K_THIRD_HEAVY_CANDIDATES="$(GTAP_K_THIRD_HEAVY_CANDIDATES)" \
	  -DGTAP_K_ENCODE_SLOTS="$(GTAP_K_ENCODE_SLOTS)" \
	  -DGTAP_K_RANGE_MAX_FIXED="$(GTAP_K_RANGE_MAX_FIXED)" \
	  -DGTAP_K_CUDA_STACK_SIZE="$(GTAP_ORIENTATION_CUDA_STACK_SIZE)" \
	  -DGTAP_DEFAULT_ORIENT="$(GTAP_DEFAULT_ORIENT)" \
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
	  -I"$(SCRIPT_DIR)" \
	  -DGTAP_K_RUNTIME_GRID_SIZE="$(GTAP_PIVOT_GRID_SIZE)" \
	  -DGTAP_K_RUNTIME_BLOCK_SIZE="$(GTAP_PIVOT_BLOCK_SIZE)" \
	  -DGTAP_K_RUNTIME_MAX_TASKS_PER_WARP="$(GTAP_PIVOT_MAX_TASKS_PER_WARP)" \
	  -DGTAP_K_RUNTIME_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  -DGTAP_K_RANGE_CUTOFF="$(GTAP_K_RANGE_CUTOFF)" \
	  -DGTAP_K_TASK_FANOUT="$(GTAP_K_TASK_FANOUT)" \
	  -DGTAP_K_MAX_CANDIDATES="$(GTAP_K_MAX_CANDIDATES)" \
	  -DGTAP_K_BIT_BUFFER_COUNT="$(GTAP_K_BIT_BUFFER_COUNT)" \
	  -DGTAP_K_ENCODE_SLOTS="$(GTAP_K_ENCODE_SLOTS)" \
	  -DGTAP_K_PIVOT_PREFIX_CHUNK="$(GTAP_K_PIVOT_PREFIX_CHUNK)" \
	  -DGTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES="$(GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES)" \
	  -DGTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES="$(GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES)" \
	  -DGTAP_K_PIVOT_PREFIX_WORK_THRESHOLD="$(GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD)" \
	  -DGTAP_K_PIVOT_PREFIX_FANOUT="$(GTAP_K_PIVOT_PREFIX_FANOUT)" \
	  -DGTAP_K_PIVOT_PREFIX_PASS_STATE="$(GTAP_K_PIVOT_PREFIX_PASS_STATE)" \
	  -DGTAP_K_RANGE_MAX_FIXED="$(GTAP_K_RANGE_MAX_FIXED)" \
	  -DGTAP_K_PIVOT_STATE_LEVELS="$(GTAP_K_PIVOT_STATE_LEVELS)" \
	  -DGTAP_K_CUDA_STACK_SIZE="$(GTAP_PIVOT_CUDA_STACK_SIZE)" \
	  -DGTAP_DEFAULT_ORIENT="$(GTAP_DEFAULT_ORIENT)" \
	  $(GTAP_CFLAGS) $(GTAP_EXTRA_CFLAGS) \
	  "$(SRC_GTAP_PIVOT)" "$(SRC_GPU_ORIENT)" -o "$(OUT_GTAP_PIVOT)" \
	  -L"$(CUDA_PATH)/lib64" -L"$(OUT_DIR)" -lcudart -lkcgpu_orient

clean:
	rm -f $(OUT_GTAP_ORIENTATION) $(OUT_GTAP_PIVOT) $(OBJ_KCGPU_ORIENT) $(LIB_KCGPU_ORIENT)
