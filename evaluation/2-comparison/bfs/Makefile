BENCHMARK_NAME := bfs

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= /work/gc64/c64099/gtap

CUDA_PATH  ?= /work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(PROJ_DIR)/clang-gtap/build/bin/clang
CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_GTAP_THREAD   := $(SCRIPT_DIR)/gtap_thread_$(BENCHMARK_NAME).cu
SRC_GTAP_BLOCK   := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME).cu

OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_GTAP_THREAD   := $(OUT_DIR)/gtap_thread_$(BENCHMARK_NAME)
OUT_GTAP_BLOCK   := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)

# ---- GTAP build settings ----
GTAP_GRID_SIZE           ?= 2000
GTAP_BLOCK_SIZE          ?= 32
GTAP_MAX_TASKS_PER_BLOCK ?= 80000
GTAP_MAX_TASKS_PER_WARP  ?= 80000
GTAP_MAX_CHILD_TASKS     ?= 8000
GTAP_NUM_QUEUES          ?= 1
GTAP_CFLAGS              ?= -DGTAP_ASSUME_NO_TASKWAIT

.PHONY: all gtap_thread gtap_block clean print-vars

all: gtap_thread gtap_block

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CLANG_BIN=$(CLANG_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Build: gtap_thread_$(BENCHMARK_NAME).cu ----
gtap_thread: $(OUT_GTAP_THREAD)

$(OUT_GTAP_THREAD): $(SRC_GTAP_THREAD) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_THREAD)" -o "$(OUT_GTAP_THREAD)"

gtap_block: $(OUT_GTAP_BLOCK)

$(OUT_GTAP_BLOCK): $(SRC_GTAP_BLOCK) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_BLOCK)" -o "$(OUT_GTAP_BLOCK)"

clean:
	rm -f $(OUT_GTAP_THREAD) $(OUT_GTAP_BLOCK)
