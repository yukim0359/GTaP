BENCHMARK_NAME := nq

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../../../common.mk

# ---- Inputs / Outputs ----
SRC_QUEUE_1   := $(SCRIPT_DIR)/$(BENCHMARK_NAME)_queue_1.cu
SRC_QUEUE_2   := $(SCRIPT_DIR)/$(BENCHMARK_NAME)_queue_2.cu

OUT_QUEUE_1   := $(OUT_DIR)/$(BENCHMARK_NAME)_queue_1
OUT_QUEUE_2   := $(OUT_DIR)/$(BENCHMARK_NAME)_queue_2

# ---- GTAP build settings ----
GTAP_GRID_SIZE          ?= 2000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 150000
GTAP_CFLAGS             ?= -fgtap-no-taskwait

.PHONY: all queue_1 queue_2 clean print-vars

all: queue_1 queue_2

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

# ---- Build: $(BENCHMARK_NAME)_queue_1.cu ----
queue_1: $(OUT_QUEUE_1)

$(OUT_QUEUE_1): $(SRC_QUEUE_1) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_QUEUE_1)" -o "$(OUT_QUEUE_1)" \
	  $(GTAP_CUDA_LIBS)

# ---- Build: $(BENCHMARK_NAME)_queue_2.cu ----
queue_2: $(OUT_QUEUE_2)

$(OUT_QUEUE_2): $(SRC_QUEUE_2) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_QUEUE_2)" -o "$(OUT_QUEUE_2)" \
	  $(GTAP_CUDA_LIBS)

clean:
	rm -f $(OUT_QUEUE_1) $(OUT_QUEUE_2)
