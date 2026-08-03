# GTaP example: SpMV (thread and block variants)
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

SRC_THREAD := $(SCRIPT_DIR)/spmv_thread.cu
SRC_BLOCK  := $(SCRIPT_DIR)/spmv_block.cu
OUT_DIR    := $(SCRIPT_DIR)/bin
OUT_THREAD := $(OUT_DIR)/spmv_thread
OUT_BLOCK  := $(OUT_DIR)/spmv_block

GTAP_CFLAGS ?=

# ---- Block variant settings ----
GTAP_BLOCK_SIZE_B ?= 128

.PHONY: all thread block clean
all: thread block

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

# ---- Thread variant ----
thread: $(OUT_THREAD)

$(OUT_THREAD): $(SRC_THREAD) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

# ---- Block variant ----
block: $(OUT_BLOCK)

$(OUT_BLOCK): $(SRC_BLOCK) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_BLOCK_SIZE=$(GTAP_BLOCK_SIZE_B) \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)"
