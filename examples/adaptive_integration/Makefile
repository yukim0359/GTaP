# GTaP example: adaptive numerical integration (thread and block variants)
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

SRC_THREAD := $(SCRIPT_DIR)/adaptive_integration_thread.cu
SRC_BLOCK  := $(SCRIPT_DIR)/adaptive_integration_block.cu
OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_THREAD := $(OUT_DIR)/adaptive_integration_thread
OUT_BLOCK  := $(OUT_DIR)/adaptive_integration_block

GTAP_CFLAGS     ?=
GTAP_BLOCK_SIZE ?= 32

.PHONY: all thread block clean
all: thread block

thread: $(OUT_THREAD)

block: $(OUT_BLOCK)

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

$(OUT_BLOCK): $(SRC_BLOCK) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_BLOCK_SIZE=$(GTAP_BLOCK_SIZE) \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)"
