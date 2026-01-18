BENCHMARK_NAME := nq

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= /work/gc64/c64099/gtap

CUDA_PATH  ?= /work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(PROJ_DIR)/clang-gtap/build/bin/clang
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_QUEUE_1   := $(SCRIPT_DIR)/$(BENCHMARK_NAME)_queue_1.cu
SRC_QUEUE_2   := $(SCRIPT_DIR)/$(BENCHMARK_NAME)_queue_2.cu

OUT_DIR       := $(SCRIPT_DIR)/bin
OUT_QUEUE_1   := $(OUT_DIR)/$(BENCHMARK_NAME)_queue_1
OUT_QUEUE_2   := $(OUT_DIR)/$(BENCHMARK_NAME)_queue_2

# ---- GTAP build settings ----
GTAP_GRID_SIZE          ?= 2000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 150000
GTAP_MAX_CHILD_TASKS    ?= 16
GTAP_CFLAGS             ?= -DGTAP_ASSUME_NO_TASKWAIT

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
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  -DGTAP_NUM_QUEUES=1 \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_QUEUE_1)" -o "$(OUT_QUEUE_1)"

# ---- Build: $(BENCHMARK_NAME)_queue_2.cu ----
queue_2: $(OUT_QUEUE_2)

$(OUT_QUEUE_2): $(SRC_QUEUE_2) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" -DGTAP_MAX_CHILD_TASKS="$(GTAP_MAX_CHILD_TASKS)" \
	  -DGTAP_NUM_QUEUES=2 \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_QUEUE_2)" -o "$(OUT_QUEUE_2)"

clean:
	rm -f $(OUT_QUEUE_1) $(OUT_QUEUE_2)
