BENCHMARK_NAME := fib

SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../../common.mk

SRC_GTAP := $(SCRIPT_DIR)/gtap_$(BENCHMARK_NAME).cu

OUT_GTAP := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)
OUT_GTAP_GQ := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)_gq
OUT_GTAP_CHASELEV := $(OUT_DIR)/gtap_$(BENCHMARK_NAME)_chaselev

GTAP_SM_COUNT           ?= 0
GTAP_GRID_SIZE          ?= 2112
GTAP_BLOCK_SIZE         ?= 32
GTAP_MAX_TASKS_PER_WARP ?= 300000
GTAP_CFLAGS             ?=

# Green Context SM partitioning needs the CUDA driver API.
GTAP_SM_LIBS := $(GTAP_CUDA_LIBS) -L"$(CUDA_PATH)/lib64/stubs" -lcuda

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
	@echo "GTAP_SM_COUNT=$(GTAP_SM_COUNT)"
	@echo "GTAP_GRID_SIZE=$(GTAP_GRID_SIZE)"
	@echo "GTAP_BLOCK_SIZE=$(GTAP_BLOCK_SIZE)"
	@echo "GTAP_MAX_TASKS_PER_WARP=$(GTAP_MAX_TASKS_PER_WARP)"

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
	  -DGTAP_SM_COUNT="$(GTAP_SM_COUNT)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP)" \
	  $(GTAP_SM_LIBS)

$(OUT_GTAP_GQ): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_SM_COUNT="$(GTAP_SM_COUNT)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DGQ \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP_GQ)" \
	  $(GTAP_SM_LIBS)

$(OUT_GTAP_CHASELEV): $(SRC_GTAP) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -Wall -Wextra \
	  -Xcuda-ptxas --warn-on-spills \
	  -I"$(GT_INC)" \
	  -DGTAP_SM_COUNT="$(GTAP_SM_COUNT)" \
	  -DGTAP_GRID_SIZE="$(GTAP_GRID_SIZE)" \
	  -DGTAP_BLOCK_SIZE="$(GTAP_BLOCK_SIZE)" \
	  -DGTAP_MAX_TASKS_PER_WARP="$(GTAP_MAX_TASKS_PER_WARP)" \
	  -DCHASELEV \
	  $(GTAP_CFLAGS) \
	  "$(SRC_GTAP)" -o "$(OUT_GTAP_CHASELEV)" \
	  $(GTAP_SM_LIBS)

clean:
	rm -f $(OUT_GTAP) $(OUT_GTAP_GQ) $(OUT_GTAP_CHASELEV)
