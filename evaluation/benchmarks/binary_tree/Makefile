BENCHMARK_NAME := binary_tree

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC_GTAP_THREAD := $(SCRIPT_DIR)/gtap_thread_$(BENCHMARK_NAME).cu
SRC_GTAP_BLOCK  := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME).cu
SRC_GTAP_BLOCK_CUTOFF := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME)_cutoff.cu

OUT_GTAP_THREAD := $(OUT_DIR)/gtap_thread_$(BENCHMARK_NAME)
OUT_GTAP_BLOCK  := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)
OUT_GTAP_BLOCK_CUTOFF := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)_cutoff

GTAP_GRID_SIZE           ?= 2000
GTAP_BLOCK_SIZE          ?= 32
GTAP_MAX_TASKS_PER_WARP  ?= 300000
GTAP_MAX_TASKS_PER_BLOCK ?= 40000
BINARY_BLOCK_CUTOFF      ?= 5

GTAP_DEFS := \
  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)"

.PHONY: all gtap_thread gtap_block gtap_block_cutoff clean print-vars

all: gtap_thread gtap_block gtap_block_cutoff

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CXX_BIN=$(CXX_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

gtap_thread: $(OUT_GTAP_THREAD)

$(OUT_GTAP_THREAD): $(SRC_GTAP_THREAD) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  $(GTAP_DEFS) \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP_THREAD)" -o "$(OUT_GTAP_THREAD)" \
	  $(GTAP_CUDA_LIBS)

gtap_block: $(OUT_GTAP_BLOCK)

$(OUT_GTAP_BLOCK): $(SRC_GTAP_BLOCK) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  $(GTAP_DEFS) \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP_BLOCK)" -o "$(OUT_GTAP_BLOCK)" \
	  $(GTAP_CUDA_LIBS)

gtap_block_cutoff: $(OUT_GTAP_BLOCK_CUTOFF)

$(OUT_GTAP_BLOCK_CUTOFF): $(SRC_GTAP_BLOCK_CUTOFF) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  $(GTAP_DEFS) \
	  -DBINARY_BLOCK_CUTOFF="$(BINARY_BLOCK_CUTOFF)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP_BLOCK_CUTOFF)" -o "$(OUT_GTAP_BLOCK_CUTOFF)" \
	  $(GTAP_CUDA_LIBS)

clean:
	rm -f $(OUT_GTAP_THREAD) $(OUT_GTAP_BLOCK) $(OUT_GTAP_BLOCK_CUTOFF)
