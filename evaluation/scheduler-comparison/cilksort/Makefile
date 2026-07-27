BENCHMARK_NAME := cilksort

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJ_DIR   ?= $(abspath $(SCRIPT_DIR)/../../..)

CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

CXX_BIN    ?= $(PROJ_DIR)/clang-gtap/build/bin/clang++
GT_INC     ?= $(PROJ_DIR)/runtime
UTIL_DIR   ?= $(PROJ_DIR)/evaluation/util

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

# ---- Data generation settings ----
DATA_SIZE   ?= 20000000
DATA_FILE   ?= $(OUT_DIR)/data_$(DATA_SIZE).bin

.PHONY: all gtap data clean print-vars

all: gtap

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CXX_BIN=$(CXX_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"
	@echo "DATA_FILE=$(DATA_FILE)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Build: gtap_$(BENCHMARK_NAME).cu ----
gtap: $(OUT_GTAP)

$(OUT_GTAP): $(SRC_GTAP) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
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

# ---- Generate test data ----
data: $(DATA_FILE)

$(DATA_FILE): | $(OUT_DIR)
	cd "$(UTIL_DIR)" && make > /dev/null 2>&1
	"$(UTIL_DIR)/gen_vector" "$(DATA_SIZE)" "$(DATA_FILE)"

# ---- Run benchmark ----
run: $(OUT_GTAP) $(DATA_FILE)
	"$(OUT_GTAP)" "$(DATA_FILE)"

clean:
	rm -f $(OUT_GTAP)
	rm -rf $(OUT_DIR)
