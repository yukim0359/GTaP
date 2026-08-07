# GTaP example: Fibonacci with profiling
GTAP_ROOT  ?= ../..
SCRIPT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CLANG_BIN  ?= $(GTAP_ROOT)/clang-gtap/build/bin/clang
GT_INC     ?= $(GTAP_ROOT)/runtime
CUDA_PATH  ?= $(CUDA_HOME)
CUDA_ARCH  ?= sm_90

OUT_DIR := $(SCRIPT_DIR)/bin
THREAD_SRC := $(SCRIPT_DIR)/fib_thread.cu
BLOCK_SRC  := $(SCRIPT_DIR)/fib_block.cu
THREAD_OUT := $(OUT_DIR)/fib_thread
BLOCK_OUT  := $(OUT_DIR)/fib_block
GTAP_BLOCK_SIZE ?= 32

GTAP_CFLAGS ?=

.PHONY: all thread block clean
all: thread block
thread: $(THREAD_OUT)
block: $(BLOCK_OUT)

$(OUT_DIR):
	mkdir -p "$(OUT_DIR)"

$(THREAD_OUT): $(THREAD_SRC) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_ENABLE_PROFILING \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(THREAD_SRC)" -o "$(THREAD_OUT)"

$(BLOCK_OUT): $(BLOCK_SRC) | $(OUT_DIR)
	"$(CLANG_BIN)" -O3 -x cuda \
	  --cuda-path="$(CUDA_PATH)" \
	  --cuda-gpu-arch="$(CUDA_ARCH)" \
	  -I"$(GT_INC)" \
	  -DGTAP_ENABLE_PROFILING \
	  -DGTAP_BLOCK_SIZE=$(GTAP_BLOCK_SIZE) \
	  $(GTAP_CFLAGS) \
	  -L"$(CUDA_PATH)/lib64" -lcudart \
	  "$(BLOCK_SRC)" -o "$(BLOCK_OUT)"

clean:
	rm -f $(THREAD_OUT) $(BLOCK_OUT)
