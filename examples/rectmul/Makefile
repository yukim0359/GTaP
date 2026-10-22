# GTaP example: rectmul (thread and block variants)
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang++
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90
MATH_LIBS  ?= $(abspath $(CUDA_PATH)/../../math_libs/lib64)
LDFLAGS    := -L"$(CUDA_PATH)/lib64" -L"$(MATH_LIBS)" \
	-Wl,-rpath,"$(CUDA_PATH)/lib64" -Wl,-rpath,"$(MATH_LIBS)" \
	-lcudart -lcublas

SRC_THREAD        := $(SCRIPT_DIR)/rectmul_thread.cu
SRC_BLOCK         := $(SCRIPT_DIR)/rectmul_block.cu
SRC_THREAD_YCHUNK := $(SCRIPT_DIR)/rectmul_thread_ychunk.cu
SRC_BLOCK_YCHUNK  := $(SCRIPT_DIR)/rectmul_block_ychunk.cu
OUT_DIR           := $(SCRIPT_DIR)/bin
OUT_THREAD        := $(OUT_DIR)/rectmul_thread
OUT_BLOCK         := $(OUT_DIR)/rectmul_block
OUT_THREAD_YCHUNK := $(OUT_DIR)/rectmul_thread_ychunk
OUT_BLOCK_YCHUNK  := $(OUT_DIR)/rectmul_block_ychunk

GTAP_CFLAGS ?=

.PHONY: all thread block thread-ychunk block-ychunk clean
all: thread block thread-ychunk block-ychunk

thread: $(OUT_THREAD)

block: $(OUT_BLOCK)

thread-ychunk: $(OUT_THREAD_YCHUNK)

block-ychunk: $(OUT_BLOCK_YCHUNK)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT_THREAD): $(SRC_THREAD) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

$(OUT_BLOCK): $(SRC_BLOCK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

$(OUT_THREAD_YCHUNK): $(SRC_THREAD_YCHUNK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  "$(SRC_THREAD_YCHUNK)" -o "$(OUT_THREAD_YCHUNK)"

$(OUT_BLOCK_YCHUNK): $(SRC_BLOCK_YCHUNK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  "$(SRC_BLOCK_YCHUNK)" -o "$(OUT_BLOCK_YCHUNK)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)" \
	  "$(OUT_THREAD_YCHUNK)" "$(OUT_BLOCK_YCHUNK)"
