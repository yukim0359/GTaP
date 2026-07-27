BENCHMARK_NAME := tree_compute_heavy

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime

# ---- Inputs / Outputs ----
SRC_GTAP_BLOCK := $(SCRIPT_DIR)/gtap_block_$(BENCHMARK_NAME).cu

OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_GTAP_BLOCK := $(OUT_DIR)/gtap_block_$(BENCHMARK_NAME)
OUT_GTAP_BLOCK_GQ := $(OUT_DIR)/gtap_block_gq_$(BENCHMARK_NAME)

# ---- GTAP build settings (block-level) ----
GTAP_GRID_SIZE            ?= 1024
GTAP_BLOCK_SIZE           ?= 64
GTAP_MAX_TASKS_PER_BLOCK  ?= 100000
GTAP_CFLAGS               ?=

# ---- Benchmark parameters ----
TREE_HEIGHT     ?= 15
COMPUTE_ITERS   ?= 1024
MEM_OPS         ?= 64

.PHONY: all gtap gtap_gq clean print-vars run

all: gtap gtap_gq

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

# ---- Build: gtap_block_$(BENCHMARK_NAME).cu (WS variant) ----
gtap: $(OUT_GTAP_BLOCK)

$(OUT_GTAP_BLOCK): $(SRC_GTAP_BLOCK) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_BLOCK)" -o "$(OUT_GTAP_BLOCK)"

# ---- Build: gtap_block_$(BENCHMARK_NAME).cu (GQ variant) ----
gtap_gq: $(OUT_GTAP_BLOCK_GQ)

$(OUT_GTAP_BLOCK_GQ): $(SRC_GTAP_BLOCK) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_BLOCK="$(GTAP_MAX_TASKS_PER_BLOCK)" \
	  -DGQ \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_GTAP_BLOCK)" -o "$(OUT_GTAP_BLOCK_GQ)"

# ---- Run benchmark ----
run: $(OUT_GTAP_BLOCK)
	"$(OUT_GTAP_BLOCK)" "$(TREE_HEIGHT)" "$(COMPUTE_ITERS)" "$(MEM_OPS)"

run_gq: $(OUT_GTAP_BLOCK_GQ)
	"$(OUT_GTAP_BLOCK_GQ)" "$(TREE_HEIGHT)" "$(COMPUTE_ITERS)" "$(MEM_OPS)"

clean:
	rm -f $(OUT_GTAP_BLOCK) $(OUT_GTAP_BLOCK_GQ)
	rm -rf $(OUT_DIR)
