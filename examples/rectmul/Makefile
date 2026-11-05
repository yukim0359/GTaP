# GTaP example: rectmul (thread and block variants)
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

CUBLAS_LIB_DIR ?= $(CUDA_PATH)/lib64

SRC_THREAD        := $(SCRIPT_DIR)/rectmul_thread.cu
SRC_BLOCK         := $(SCRIPT_DIR)/rectmul_block.cu
SRC_THREAD_YCHUNK := $(SCRIPT_DIR)/rectmul_thread_ychunk.cu
SRC_BLOCK_YCHUNK  := $(SCRIPT_DIR)/rectmul_block_ychunk.cu
OUT_THREAD        := $(OUT_DIR)/rectmul_thread
OUT_BLOCK         := $(OUT_DIR)/rectmul_block
OUT_THREAD_YCHUNK := $(OUT_DIR)/rectmul_thread_ychunk
OUT_BLOCK_YCHUNK  := $(OUT_DIR)/rectmul_block_ychunk

GTAP_CFLAGS ?=
LDFLAGS ?=
LDLIBS  ?=

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
	  $(GTAP_CUDA_LIBS) -L"$(CUBLAS_LIB_DIR)" \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)" \
	  -lcublas $(LDLIBS)

$(OUT_BLOCK): $(SRC_BLOCK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  $(GTAP_CUDA_LIBS) -L"$(CUBLAS_LIB_DIR)" \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)" \
	  -lcublas $(LDLIBS)

$(OUT_THREAD_YCHUNK): $(SRC_THREAD_YCHUNK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  $(GTAP_CUDA_LIBS) -L"$(CUBLAS_LIB_DIR)" \
	  "$(SRC_THREAD_YCHUNK)" -o "$(OUT_THREAD_YCHUNK)" \
	  -lcublas $(LDLIBS)

$(OUT_BLOCK_YCHUNK): $(SRC_BLOCK_YCHUNK) $(SCRIPT_DIR)/rectmul_host.hpp | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(LDFLAGS) \
	  $(GTAP_CUDA_LIBS) -L"$(CUBLAS_LIB_DIR)" \
	  "$(SRC_BLOCK_YCHUNK)" -o "$(OUT_BLOCK_YCHUNK)" \
	  -lcublas $(LDLIBS)

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)" \
	  "$(OUT_THREAD_YCHUNK)" "$(OUT_BLOCK_YCHUNK)"
