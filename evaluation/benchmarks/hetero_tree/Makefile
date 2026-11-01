SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC_THREAD     := $(SCRIPT_DIR)/gtap_thread_hetero_tree.cu
SRC_THREAD_DAQ := $(SCRIPT_DIR)/gtap_thread_hetero_tree_daq.cu
SRC_BLOCK      := $(SCRIPT_DIR)/gtap_block_hetero_tree.cu
SRC_BLOCK_CUTOFF := $(SCRIPT_DIR)/gtap_block_hetero_tree_cutoff.cu
COMMON_HDR     := $(SCRIPT_DIR)/hetero_tree_common.cuh

OUT_THREAD     := $(OUT_DIR)/gtap_thread_hetero_tree
OUT_THREAD_DAQ := $(OUT_DIR)/gtap_thread_hetero_tree_daq
OUT_BLOCK      := $(OUT_DIR)/gtap_block_hetero_tree
OUT_BLOCK_CUTOFF := $(OUT_DIR)/gtap_block_hetero_tree_cutoff

HETERO_TREE_K         ?= 2
HETERO_DAQ_NUM_QUEUES ?= $(shell expr $(HETERO_TREE_K) + 1)
HETERO_BLOCK_CUTOFF   ?= 5

GTAP_GRID_SIZE           ?= 2000
GTAP_BLOCK_SIZE          ?= 32
GTAP_MAX_TASKS_PER_WARP  ?= 300000
GTAP_MAX_TASKS_PER_BLOCK ?= 300000
GTAP_CFLAGS              ?=

# Profile builds: make GTAP_ENABLE_PROFILING=1
GTAP_ENABLE_PROFILING ?= 0
ifeq ($(GTAP_ENABLE_PROFILING),1)
GTAP_CFLAGS += -DGTAP_ENABLE_PROFILING
endif

.PHONY: all thread thread_daq block block_cutoff clean print-vars \
        sweep-compute sweep-compute-k24

all: thread thread_daq block block_cutoff

sweep-compute:
	HETERO_TREE_K="$(HETERO_TREE_K)" bash "$(SCRIPT_DIR)/compare_hetero_compute.sh"

sweep-compute-k24:
	bash "$(SCRIPT_DIR)/compare_hetero_compute_k24.sh"

print-vars:
	@echo "HETERO_TREE_K=$(HETERO_TREE_K)"
	@echo "HETERO_DAQ_NUM_QUEUES=$(HETERO_DAQ_NUM_QUEUES)"
	@echo "HETERO_BLOCK_CUTOFF=$(HETERO_BLOCK_CUTOFF)"
	@echo "GTAP_ENABLE_PROFILING=$(GTAP_ENABLE_PROFILING)"
	@echo "OUT_DIR=$(OUT_DIR)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

thread: $(OUT_THREAD)

$(OUT_THREAD): $(SRC_THREAD) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)" \
	  $(GTAP_CUDA_LIBS)

thread_daq: $(OUT_THREAD_DAQ)

$(OUT_THREAD_DAQ): $(SRC_THREAD_DAQ) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DHETERO_DAQ_NUM_QUEUES="$(HETERO_DAQ_NUM_QUEUES)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_THREAD_DAQ)" -o "$(OUT_THREAD_DAQ)" \
	  $(GTAP_CUDA_LIBS)

block: $(OUT_BLOCK)

$(OUT_BLOCK): $(SRC_BLOCK) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BENCH_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)" \
	  $(GTAP_CUDA_LIBS)

block_cutoff: $(OUT_BLOCK_CUTOFF)

$(OUT_BLOCK_CUTOFF): $(SRC_BLOCK_CUTOFF) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BENCH_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  -DHETERO_BLOCK_CUTOFF="$(HETERO_BLOCK_CUTOFF)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_BLOCK_CUTOFF)" -o "$(OUT_BLOCK_CUTOFF)" \
	  $(GTAP_CUDA_LIBS)

clean:
	rm -f $(OUT_THREAD) $(OUT_THREAD_DAQ) $(OUT_BLOCK) $(OUT_BLOCK_CUTOFF)
