BENCHMARK_NAME := nq

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CLANG_BIN  ?= $(PROJ_DIR)/clang-gtap/build/bin/clang
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_GTAP   := $(SCRIPT_DIR)/gtap_$(BENCHMARK_NAME).cu

OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_GTAP   := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)

# ---- GTAP build settings ----
GTAP_GRID_SIZE          ?= 4000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 100000
GTAP_NUM_QUEUES         ?= 1
GTAP_CFLAGS             ?=

# ---- N-Queens parameters ----
NQ_GRID_SIZE    ?= 16
NQ_CUTOFF_DEPTH ?= 7

.PHONY: all gtap clean print-vars run

all: gtap

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

# ---- Build: gtap_$(BENCHMARK_NAME).cu ----
gtap: $(OUT_GTAP)

$(OUT_GTAP): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP)"

# ---- Run benchmark ----
run: $(OUT_GTAP)
	"$(OUT_GTAP)" "$(NQ_GRID_SIZE)" "$(NQ_CUTOFF_DEPTH)"

clean:
	rm -f $(OUT_GTAP)
	rm -rf $(OUT_DIR)


