BENCHMARK_NAME := fib

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../../common.mk

SRC_GTAP   := $(SCRIPT_DIR)/gtap_$(BENCHMARK_NAME).cu
SRC_OMP    := $(SCRIPT_DIR)/omp_$(BENCHMARK_NAME).c
SRC_CILK   := $(SCRIPT_DIR)/cilk_$(BENCHMARK_NAME).c

OUT_GTAP   := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)
OUT_OMP    := $(OUT_DIR)/omp_$(BENCHMARK_NAME)
OUT_CILK   := $(OUT_DIR)/cilk_$(BENCHMARK_NAME)

GTAP_GRID_SIZE          ?= 3000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 120000
GTAP_NUM_QUEUES         ?= 1

.PHONY: all gtap omp cilk clean print-vars

all: gtap omp cilk

print-vars:
	@echo "SCRIPT_DIR=$(SCRIPT_DIR)"
	@echo "PROJ_DIR=$(PROJ_DIR)"
	@echo "CLANG_BIN=$(CLANG_BIN)"
	@echo "CUDA_PATH=$(CUDA_PATH)"
	@echo "CUDA_ARCH=$(CUDA_ARCH)"
	@echo "GT_INC=$(GT_INC)"
	@echo "OUT_DIR=$(OUT_DIR)"
	@echo "OPENCILK_PREFIX=$(OPENCILK_PREFIX)"
	@echo "CC_CILK=$(CC_CILK)"

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

gtap: $(OUT_GTAP)

$(OUT_GTAP): $(SRC_GTAP) | $(OUT_DIR)
	"$(CXX_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas -v \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DGTAP_NUM_QUEUES="$(GTAP_NUM_QUEUES)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP)" \
	  $(GTAP_CUDA_LIBS)

omp: $(OUT_OMP)

$(OUT_OMP): $(SRC_OMP) | $(OUT_DIR)
	"$(CC_OMP)" $(CFLAGS_OMP) "$(SRC_OMP)" -o "$(OUT_OMP)" $(LDFLAGS_OMP)

cilk: $(OUT_CILK)

$(OUT_CILK): $(SRC_CILK) | $(OUT_DIR)
	@test -x "$(CC_CILK)" || (echo "Error: OpenCilk not found at $(CC_CILK). Run: $(PROJ_DIR)/evaluation/scripts/build_opencilk.sh" && false)
	"$(CC_CILK)" $(CFLAGS_CILK) "$(SRC_CILK)" -o "$(OUT_CILK)" $(LDFLAGS_CILK)

clean:
	rm -f $(OUT_GTAP) $(OUT_OMP) $(OUT_CILK)
