# GTaP example: Fibonacci
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SCRIPT_DIR)/../common.mk

SRC_THREAD := $(SCRIPT_DIR)/fib_thread.cu
SRC_BLOCK  := $(SCRIPT_DIR)/fib_block.cu
OUT_THREAD := $(OUT_DIR)/fib_thread
OUT_BLOCK  := $(OUT_DIR)/fib_block
OUT_THREAD_PROFILE := $(OUT_DIR)/fib_thread_profile
OUT_BLOCK_PROFILE  := $(OUT_DIR)/fib_block_profile

GTAP_CFLAGS ?=

.PHONY: all thread block profile profile-thread profile-block clean
all: thread block

thread: $(OUT_THREAD)

block: $(OUT_BLOCK)

profile: profile-thread profile-block

profile-thread: $(OUT_THREAD_PROFILE)

profile-block: $(OUT_BLOCK_PROFILE)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(OUT_THREAD): $(SRC_THREAD) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD)"

$(OUT_BLOCK): $(SRC_BLOCK) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK)"

$(OUT_THREAD_PROFILE): $(SRC_THREAD) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_ENABLE_PROFILING \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_THREAD)" -o "$(OUT_THREAD_PROFILE)"

$(OUT_BLOCK_PROFILE): $(SRC_BLOCK) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_ENABLE_PROFILING \
	  $(GTAP_CFLAGS) \
	  $(GTAP_CUDA_LIBS) \
	  "$(SRC_BLOCK)" -o "$(OUT_BLOCK_PROFILE)"

clean:
	rm -f "$(OUT_THREAD)" "$(OUT_BLOCK)" \
	  "$(OUT_THREAD_PROFILE)" "$(OUT_BLOCK_PROFILE)"
