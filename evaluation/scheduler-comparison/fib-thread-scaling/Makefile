BENCHMARK_NAME := fib

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../../common.mk

SRC_GTAP := $(SCRIPT_DIR)/gtap_$(BENCHMARK_NAME).cu

OUT_GTAP := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)
OUT_GTAP_GQ := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)_gq
OUT_GTAP_CHASELEV := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)_chaselev

GTAP_GRID_SIZE          ?= 4000
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 200000
GTAP_CFLAGS             ?=

.PHONY: all gtap gtap_gq gtap_chaselev clean print-vars

all: gtap gtap_gq gtap_chaselev

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

gtap: $(OUT_GTAP)
gtap_gq: $(OUT_GTAP_GQ)
gtap_chaselev: $(OUT_GTAP_CHASELEV)

$(OUT_GTAP): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP)" \
	  $(GTAP_CUDA_LIBS)

$(OUT_GTAP_GQ): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DGQ \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP_GQ)" \
	  $(GTAP_CUDA_LIBS)

$(OUT_GTAP_CHASELEV): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DCHASELEV \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP_CHASELEV)" \
	  $(GTAP_CUDA_LIBS)

clean:
	rm -f $(OUT_GTAP) $(OUT_GTAP_GQ) $(OUT_GTAP_CHASELEV)
