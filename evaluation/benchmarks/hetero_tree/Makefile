SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

OUT_DIR := $(SCRIPT_DIR)/bin

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

# Profile builds: make GTAP_PROFILE=1 ...  or  make profile
GTAP_PROFILE ?= 0
ifeq ($(GTAP_PROFILE),1)
GTAP_CFLAGS += -DGTAP_PROFILE
endif

.PHONY: all thread thread_daq block block_cutoff clean print-vars \
        sweep-depth sweep-compute sweep-compute-k24 plot profile run-profile plot-profile

all: thread thread_daq block block_cutoff

# Force rebuild of all four with -DGTAP_PROFILE
profile:
	$(MAKE) -B all GTAP_PROFILE=1

# PBS / interactive sweeps (override NUM_RUNS / DEPTH / COMPUTE via env)
sweep-depth: all
	bash "$(SCRIPT_DIR)/compare_hetero_depth.sh"

sweep-compute:
	HETERO_TREE_K="$(HETERO_TREE_K)" bash "$(SCRIPT_DIR)/compare_hetero_compute.sh"

# K=2 then K=4 compute sweeps + plots (long; prefer qsub)
sweep-compute-k24:
	bash "$(SCRIPT_DIR)/compare_hetero_compute_k24.sh"

plot:
	cd "$(SCRIPT_DIR)" && python3 plot_performance_hetero_tree.py

run-profile: profile
	bash "$(SCRIPT_DIR)/run_hetero_profiles.sh"

plot-profile:
	python3 "$(SCRIPT_DIR)/../../scripts/plot_hetero_tree_timeline_paper.py"

print-vars:
	@echo "HETERO_TREE_K=$(HETERO_TREE_K)"
	@echo "HETERO_DAQ_NUM_QUEUES=$(HETERO_DAQ_NUM_QUEUES)"
	@echo "HETERO_BLOCK_CUTOFF=$(HETERO_BLOCK_CUTOFF)"
	@echo "GTAP_PROFILE=$(GTAP_PROFILE)"
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
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

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
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_THREAD_DAQ)" -o "$(OUT_THREAD_DAQ)"

block: $(OUT_BLOCK)

$(OUT_BLOCK): $(SRC_BLOCK) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

block_cutoff: $(OUT_BLOCK_CUTOFF)

$(OUT_BLOCK_CUTOFF): $(SRC_BLOCK_CUTOFF) $(COMMON_HDR) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" -I"$(SCRIPT_DIR)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_BENCH_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BENCH_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  -DHETERO_TREE_K="$(HETERO_TREE_K)" \
	  -DHETERO_BLOCK_CUTOFF="$(HETERO_BLOCK_CUTOFF)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_BLOCK_CUTOFF)" -o "$(OUT_BLOCK_CUTOFF)"

clean:
	rm -f $(OUT_THREAD) $(OUT_THREAD_DAQ) $(OUT_BLOCK) $(OUT_BLOCK_CUTOFF)
