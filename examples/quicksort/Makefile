# GTaP example: Quicksort (thread and block variants)
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

SRC_THREAD   := $(SCRIPT_DIR)/quicksort_thread.cu
SRC_ADVANCED := $(SCRIPT_DIR)/quicksort_advanced_block.cu
OUT_DIR      := $(SCRIPT_DIR)/bin
OUT_THREAD   := $(OUT_DIR)/quicksort_thread
OUT_ADVANCED := $(OUT_DIR)/quicksort_advanced_block
OUT_ADVANCED_PROFILE := $(OUT_DIR)/quicksort_advanced_block_profile

GTAP_CFLAGS ?=

.PHONY: all thread block profile clean
all: thread block

thread: $(OUT_THREAD)

block: $(OUT_ADVANCED)

profile: $(OUT_ADVANCED_PROFILE)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT_THREAD): $(SRC_THREAD) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

$(OUT_ADVANCED): $(SRC_ADVANCED) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_ADVANCED)" -o "$(OUT_ADVANCED)"

$(OUT_ADVANCED_PROFILE): $(SRC_ADVANCED) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_ENABLE_PROFILING \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_ADVANCED)" -o "$(OUT_ADVANCED_PROFILE)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_ADVANCED)" "$(OUT_ADVANCED_PROFILE)"
