# GTAP block pivot k-clique
#
# Same pivot algorithm + prefix task splitting as ../gtap_pivot (thread runtime),
# but uses gtap_block with KCGPU-style cooperative bit ops inside each CUDA block.
#
# Compare:
#   ../bin/gtap_pivot          — thread runtime, serial bit ops, prefix ON
#   bin/gtap_pivot_block       — block runtime, cooperative bit ops, prefix ON
#   KCGPU pivot                — no task split, block-cooperative bit ops

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
K_CLIQUE_DIR := $(abspath $(SCRIPT_DIR)/..)
PROJ_DIR   ?= $(abspath $(K_CLIQUE_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

SRC := $(SCRIPT_DIR)/gtap_k_clique_pivot_block.cu
SRC_GPU_ORIENT := $(K_CLIQUE_DIR)/k_clique_gpu_orient.cu
OUT_DIR := $(SCRIPT_DIR)/bin
OUT := $(OUT_DIR)/gtap_pivot_block

# Block pivot tuning: more scheduler workers + coarser prefix split (one edgek_edge per edge).
# Task pool memory ~ task_stride(516) × GRID × MAX_TASKS_PER_BLOCK (+ ~10% headers).
GTAP_PIVOT_BLOCK_GRID_SIZE          ?= 4000
GTAP_PIVOT_BLOCK_BLOCK_SIZE         ?= 128
GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK  ?= 10000

GTAP_K_RANGE_CUTOFF ?= 256
GTAP_K_TASK_FANOUT ?= 128
GTAP_K_MAX_CANDIDATES ?= 512
GTAP_K_BIT_BUFFER_COUNT ?= 256
GTAP_K_ENCODE_SLOTS ?= 4096
# Wider chunks / higher bars -> fewer prefix subtasks (block has only GRID workers).
GTAP_K_PIVOT_PREFIX_CHUNK ?= 64
GTAP_K_PIVOT_SHARED_SPLIT_CANDIDATES ?= 512
GTAP_K_PIVOT_PREFIX_HEAVY_CANDIDATES ?= 64
GTAP_K_PIVOT_PREFIX_WORK_THRESHOLD ?= 65536
GTAP_K_PIVOT_PREFIX_FANOUT ?= 32
GTAP_K_BLOCK_ENCODE_KCGPU_STYLE ?= 1
GTAP_K_RANGE_MAX_FIXED ?= 24
GTAP_K_PIVOT_STATE_LEVELS ?= 128
GTAP_K_CUDA_STACK_SIZE ?= 4096
GTAP_CFLAGS ?=
GTAP_EXTRA_CFLAGS ?=
GTAP_ORIENT ?= degen
ifeq ($(GTAP_ORIENT),degree)
GTAP_DEFAULT_ORIENT := GTAP_ORIENT_DEGREE
else ifeq ($(GTAP_ORIENT),degen)
GTAP_DEFAULT_ORIENT := GTAP_ORIENT_DEGEN
else
$(error GTAP_ORIENT must be degree or degen, got "$(GTAP_ORIENT)")
endif
GTAP_K_STATS ?= 0
ifeq ($(GTAP_K_STATS),1)
GTAP_EXTRA_CFLAGS += -DGTAP_K_STATS
endif
GTAP_PROFILE ?= 0
ifeq ($(GTAP_PROFILE),1)
GTAP_EXTRA_CFLAGS += -DPROFILE
else
GTAP_EXTRA_CFLAGS += -UPROFILE
endif

.PHONY: all clean print-vars

all: $(OUT)

print-vars:
	@echo "OUT=$(OUT)"
	@echo "GTAP_PIVOT_BLOCK_GRID_SIZE=$(GTAP_PIVOT_BLOCK_GRID_SIZE)"
	@echo "GTAP_PIVOT_BLOCK_BLOCK_SIZE=$(GTAP_PIVOT_BLOCK_BLOCK_SIZE)"
	@echo "GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK=$(GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK)"
	@echo "GTAP_K_BLOCK_ENCODE_KCGPU_STYLE=$(GTAP_K_BLOCK_ENCODE_KCGPU_STYLE)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT): $(SRC) $(SRC_GPU_ORIENT) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -I"$(SCRIPT_DIR)" \
	  -I"$(K_CLIQUE_DIR)" \
	  -DGTAP_GRID_SIZE="$(GTAP_PIVOT_BLOCK_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_PIVOT_BLOCK_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK)" \
	  -DGTAP_PIVOT_BLOCK_GRID_SIZE="$(GTAP_PIVOT_BLOCK_GRID_SIZE)" \
	  -DGTAP_PIVOT_BLOCK_BLOCK_SIZE="$(GTAP_PIVOT_BLOCK_BLOCK_SIZE)" \
	  -DGTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK="$(GTAP_PIVOT_BLOCK_MAX_TASKS_PER_BLOCK)" \
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
	  -DGTAP_K_BLOCK_ENCODE_KCGPU_STYLE="$(GTAP_K_BLOCK_ENCODE_KCGPU_STYLE)" \
	  -DGTAP_K_RANGE_MAX_FIXED="$(GTAP_K_RANGE_MAX_FIXED)" \
	  -DGTAP_K_PIVOT_STATE_LEVELS="$(GTAP_K_PIVOT_STATE_LEVELS)" \
	  -DGTAP_K_CUDA_STACK_SIZE="$(GTAP_K_CUDA_STACK_SIZE)" \
	  -DGTAP_DEFAULT_ORIENT="$(GTAP_DEFAULT_ORIENT)" \
	  $(GTAP_CFLAGS) $(GTAP_EXTRA_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC)" "$(SRC_GPU_ORIENT)" -o "$(OUT)"

clean:
	rm -f "$(OUT)"
